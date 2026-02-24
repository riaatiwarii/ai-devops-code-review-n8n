"""
CodeReviewModel — wraps HuggingFace inference for a fine-tuned code LLM.
Supports: DeepSeek-Coder, Code Llama, StarCoder.
"""

import json
import logging
import re
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from schemas import DiffInput, ReviewOutput, CodeIssue

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert code reviewer. Analyze the following code diff and return a JSON object with this exact schema:
{
  "issues": [
    {
      "file": "<filename>",
      "line": <line_number>,
      "severity": "<low|medium|high|critical>",
      "type": "<bug|security|performance|style|maintainability>",
      "description": "<what is wrong>",
      "suggested_fix": "<how to fix it>",
      "confidence": <0.0-1.0>
    }
  ],
  "overall_score": <0.0-10.0>,
  "summary": "<one paragraph summary>"
}
Return ONLY valid JSON. No markdown, no explanation."""


class CodeReviewModel:
    def __init__(self, model_name: str, device: str = "cuda", lora_adapter: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.lora_adapter = lora_adapter
        self.tokenizer = None
        self.model = None
        self.is_ready = False
        self.version = "base"

    async def load(self):
        """Load model + optional LoRA adapter."""
        logger.info(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device == "cuda":
            # 4-bit quantization — GPU only
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            logger.info("Loading model with 4-bit quantization (GPU)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # CPU mode — no quantization, uses float32
            logger.info("Loading model on CPU (takes 1-3 mins on first run, normal)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

        # Load LoRA fine-tuned adapter if provided
        if self.lora_adapter:
            logger.info(f"Loading LoRA adapter: {self.lora_adapter}")
            self.model = PeftModel.from_pretrained(self.model, self.lora_adapter)
            self.version = "fine-tuned"

        self.model.eval()
        self.is_ready = True
        logger.info(f"Model ready ({self.version})")

    def _build_prompt(self, diff_input: DiffInput) -> str:
        user_content = (
            f"Language: {diff_input.language}\n"
            f"Files changed: {', '.join(diff_input.files)}\n"
            f"Lines added: {diff_input.lines_added} | Lines removed: {diff_input.lines_removed}\n\n"
            f"=== DIFF ===\n{diff_input.diff[:6000]}\n=== END DIFF ==="
        )
        # DeepSeek-Coder chat format
        return (
            f"<|system|>\n{SYSTEM_PROMPT}\n<|end|>\n"
            f"<|user|>\n{user_content}\n<|end|>\n"
            f"<|assistant|>\n"
        )

    async def review(self, diff_input: DiffInput) -> ReviewOutput:
        prompt = self._build_prompt(diff_input)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,  # Reduced for CPU speed
                temperature=0.1,          # Low temp → deterministic, less hallucination
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return self._parse_output(generated)

    def _parse_output(self, raw: str) -> ReviewOutput:
        """Robustly parse model output into ReviewOutput."""
        # Try to extract JSON block if model wrapped it
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            raw = json_match.group(0)

        try:
            data = json.loads(raw)
            issues = [CodeIssue(**i) for i in data.get("issues", [])]
            return ReviewOutput(
                issues=issues,
                overall_score=float(data.get("overall_score", 7.0)),
                summary=data.get("summary"),
                model_version=self.version,
            )
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse model output: {e}\nRaw: {raw[:200]}")
            # Fallback: return empty review with low score to flag manual review
            return ReviewOutput(
                issues=[],
                overall_score=5.0,
                summary=f"Automated review parse error — manual review recommended. Raw: {raw[:300]}",
                model_version=self.version,
            )