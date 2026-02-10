import asyncio
import logging
from datetime import datetime
from typing import Dict, Type

from src.common.config import get_settings
from src.ingestion.adapters.base import DataSourceAdapter
from src.ingestion.embedder import Embedder, SparseEncoder
from src.ingestion.processor import Processor
from src.ingestion.sync_state import SyncStateDB
from src.ingestion.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionScheduler:
    """
    백그라운드에서 주기적으로 데이터를 수집하는 스케줄러.
    asyncio 태스크로 동작하며, SyncStateDB를 통해 증분 수집을 수행합니다.
    각 어댑터별로 독립적인 수집 주기를 지원합니다.
    """

    def __init__(
        self,
        adapters: Dict[str, Type[DataSourceAdapter]],
        processor: Processor,
        embedder: Embedder,
        sparse_encoder: SparseEncoder,
        vector_store: VectorStore,
    ):
        self._adapters = adapters
        self._processor = processor
        self._embedder = embedder
        self._sparse_encoder = sparse_encoder
        self._store = vector_store
        self._sync_db = SyncStateDB()
        self._task: asyncio.Task | None = None
        self._running = False
        self._last_run: Dict[str, datetime] = {}  # 어댑터별 마지막 실행 시간

    def _get_adapter_interval(self, source_name: str) -> int:
        """어댑터별 수집 주기(분)를 반환합니다. 0이면 글로벌 설정 사용."""
        settings = get_settings()
        adapter_intervals = {
            "slack": settings.slack_sync_interval_minutes,
            "jira": settings.jira_sync_interval_minutes,
        }
        interval = adapter_intervals.get(source_name, 0)
        return interval if interval > 0 else settings.ingestion_interval_minutes

    def _should_run(self, source_name: str) -> bool:
        """해당 어댑터가 실행되어야 하는지 확인합니다."""
        if source_name not in self._last_run:
            return True

        interval_minutes = self._get_adapter_interval(source_name)
        elapsed = (datetime.utcnow() - self._last_run[source_name]).total_seconds()
        return elapsed >= interval_minutes * 60

    async def start(self):
        """스케줄러를 시작합니다."""
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Ingestion 스케줄러 시작")

    async def stop(self):
        """스케줄러를 중지합니다."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Ingestion 스케줄러 중지")

    async def _loop(self):
        """주기적으로 소스별 수집 주기에 따라 수집을 실행합니다."""
        # 최초 기동 시 짧은 대기 후 바로 1회 실행
        await asyncio.sleep(10)

        while self._running:
            for source_name, adapter_cls in self._adapters.items():
                if not self._should_run(source_name):
                    continue

                try:
                    logger.info(f"[{source_name}] 스케줄링된 Ingestion 시작")
                    await self._ingest_source(source_name, adapter_cls)
                    self._last_run[source_name] = datetime.utcnow()
                    interval = self._get_adapter_interval(source_name)
                    logger.info(f"[{source_name}] Ingestion 완료. 다음 실행까지 {interval}분")
                except Exception:
                    logger.exception(f"[{source_name}] 스케줄링 Ingestion 실패")

            # 1분마다 체크 (각 어댑터별 개별 주기 확인)
            await asyncio.sleep(60)

    async def _ingest_source(self, source_name: str, adapter_cls: Type[DataSourceAdapter]):
        """단일 소스에 대한 증분 수집을 수행합니다."""
        last_sync = self._sync_db.get_last_sync(source_name)
        sync_start = datetime.utcnow()

        if last_sync:
            logger.info(f"[{source_name}] 증분 수집 (since {last_sync.isoformat()})")
        else:
            logger.info(f"[{source_name}] 전체 수집 (최초 실행)")

        # blocking I/O를 별도 스레드에서 실행
        adapter = adapter_cls()
        raw_documents = await asyncio.to_thread(adapter.fetch_updates, last_sync)

        if not raw_documents:
            logger.info(f"[{source_name}] 새 문서 없음, 건너뜀")
            self._sync_db.update_last_sync(source_name, sync_start)
            return

        parents, children = await asyncio.to_thread(self._processor.process, raw_documents)

        if not children:
            logger.info(f"[{source_name}] 유효한 청크 없음, 건너뜀")
            self._sync_db.update_last_sync(source_name, sync_start)
            return

        dense_vectors = await asyncio.to_thread(self._embedder.generate_embeddings, children)

        self._sparse_encoder.fit([c.content for c in children])
        sparse_vectors = await asyncio.to_thread(self._sparse_encoder.encode_batch, children)

        await asyncio.to_thread(self._store.upsert_parents, parents)
        await asyncio.to_thread(self._store.upsert_children, children, dense_vectors, sparse_vectors)

        self._sync_db.update_last_sync(source_name, sync_start)

        logger.info(
            f"[{source_name}] 수집 완료: 문서 {len(raw_documents)}개 → "
            f"Parent {len(parents)}개, Child {len(children)}개"
        )
