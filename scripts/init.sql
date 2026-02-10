-- RAG Service Database Initialization
-- This script is automatically executed when MySQL container starts for the first time.

-- Create sync_state table for incremental ingestion
CREATE TABLE IF NOT EXISTS sync_state (
    source_id VARCHAR(255) PRIMARY KEY COMMENT '데이터 소스 식별자 (slack, jira 등)',
    last_sync_at VARCHAR(255) NOT NULL COMMENT '마지막 동기화 시점 (ISO 8601 format)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '레코드 생성 시점',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '레코드 수정 시점'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_sync_state_updated_at ON sync_state(updated_at);
