#!/bin/bash
# Create multiple databases for n8n and code_reviews
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE code_reviews;
    CREATE DATABASE n8n;
    GRANT ALL PRIVILEGES ON DATABASE code_reviews TO $POSTGRES_USER;
    GRANT ALL PRIVILEGES ON DATABASE n8n TO $POSTGRES_USER;
EOSQL