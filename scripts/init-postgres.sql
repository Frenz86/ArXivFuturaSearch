-- =============================================================================
-- ArXiv RAG Copilot - PostgreSQL Initialization Script
-- =============================================================================
-- This script is automatically run when the PostgreSQL container starts
-- for the first time. It installs and configures the pgvector extension.
-- =============================================================================

-- Install pgvector extension (if not already installed)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create indexes for better query performance
-- Note: These indexes will be created by LangChain PGVector automatically
-- when documents are added. We keep this file for reference and future enhancements.

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE ${POSTGRES_DB} TO ${POSTGRES_USER};

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'ArXiv RAG Copilot: PostgreSQL initialized with pgvector extension';
END $$;
