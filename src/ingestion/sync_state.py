import logging
from datetime import datetime
from typing import Optional

import pymysql

from src.common.config import get_settings

logger = logging.getLogger(__name__)


class SyncStateDB:
    """
    소스별 마지막 동기화 시점을 관리하는 MySQL DB.
    증분 수집(Incremental Ingestion)을 위해 사용됩니다.
    """

    def __init__(self):
        settings = get_settings()
        self._conn_params = {
            "host": settings.mysql_host,
            "port": settings.mysql_port,
            "user": settings.mysql_user,
            "password": settings.mysql_password,
            "database": settings.mysql_database,
            "charset": "utf8mb4",
        }
        self._init_table()

    def _get_conn(self) -> pymysql.Connection:
        return pymysql.connect(**self._conn_params)

    def _init_table(self):
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sync_state (
                        source_id VARCHAR(255) PRIMARY KEY,
                        last_sync_at VARCHAR(255) NOT NULL
                    )
                    """
                )
            conn.commit()

    def get_last_sync(self, source_id: str) -> Optional[datetime]:
        """소스의 마지막 동기화 시점을 조회합니다."""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT last_sync_at FROM sync_state WHERE source_id = %s",
                    (source_id,),
                )
                row = cur.fetchone()
        if row:
            return datetime.fromisoformat(row[0])
        return None

    def update_last_sync(self, source_id: str, timestamp: Optional[datetime] = None):
        """소스의 마지막 동기화 시점을 갱신합니다."""
        ts = (timestamp or datetime.utcnow()).isoformat()
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sync_state (source_id, last_sync_at)
                    VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE last_sync_at = VALUES(last_sync_at)
                    """,
                    (source_id, ts),
                )
            conn.commit()
        logger.info(f"[{source_id}] 동기화 시점 갱신: {ts}")
