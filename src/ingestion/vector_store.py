import logging
from typing import Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    NamedSparseVector,
    NamedVector,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.common.config import get_settings
from src.ingestion.processor import ChildChunk, ParentChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Qdrant Vector DB 커넥터.
    Parent/Child 컬렉션을 관리하고 벡터를 Upsert합니다.
    """

    def __init__(self):
        settings = get_settings()
        self._client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self._child_collection = settings.qdrant_child_collection
        self._parent_collection = settings.qdrant_parent_collection
        self._embedding_dim = settings.openai_embedding_dim

    def init_collections(self):
        """Child 및 Parent 컬렉션이 없으면 생성합니다."""
        self._ensure_child_collection()
        self._ensure_parent_collection()

    def _ensure_child_collection(self):
        """Child Chunk 컬렉션: Dense + Sparse Named Vector 지원."""
        collections = [c.name for c in self._client.get_collections().collections]
        if self._child_collection not in collections:
            self._client.create_collection(
                collection_name=self._child_collection,
                vectors_config={
                    "dense": VectorParams(
                        size=self._embedding_dim,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(),
                },
            )
            logger.info(f"'{self._child_collection}' 컬렉션 생성 완료")

    def _ensure_parent_collection(self):
        """Parent Chunk 컬렉션: 텍스트 저장 전용 (벡터 없음, ID 기반 조회)."""
        collections = [c.name for c in self._client.get_collections().collections]
        if self._parent_collection not in collections:
            self._client.create_collection(
                collection_name=self._parent_collection,
                vectors_config={
                    "dummy": VectorParams(size=1, distance=Distance.COSINE),
                },
            )
            logger.info(f"'{self._parent_collection}' 컬렉션 생성 완료")

    def upsert_parents(self, parents: List[ParentChunk]):
        """Parent Chunk를 Qdrant에 저장합니다 (ID 기반 조회용)."""
        points = [
            PointStruct(
                id=parent.chunk_id,
                vector={"dummy": [0.0]},
                payload={
                    "doc_id": parent.doc_id,
                    "content": parent.content,
                    "source": parent.source,
                    "url": parent.url,
                    "author": parent.author,
                    "metadata": parent.metadata,
                },
            )
            for parent in parents
        ]
        self._batch_upsert(self._parent_collection, points)
        logger.info(f"Parent {len(parents)}개 Upsert 완료")

    def upsert_children(
        self,
        children: List[ChildChunk],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Tuple[List[int], List[float]]],
    ):
        """Child Chunk를 Dense + Sparse Vector와 함께 저장합니다."""
        points = []
        for child, dense, (sparse_indices, sparse_values) in zip(
            children, dense_vectors, sparse_vectors
        ):
            acl = child.permissions
            point = PointStruct(
                id=child.chunk_id,
                vector={
                    "dense": dense,
                    "sparse": SparseVector(indices=sparse_indices, values=sparse_values),
                },
                payload={
                    "parent_id": child.parent_id,
                    "content_text": child.content,
                    "source": child.source,
                    "url": child.url,
                    "author": child.author,
                    "created_at": (
                        child.metadata.get("created_at", "") if child.metadata else ""
                    ),
                    "metadata": child.metadata,
                    "permissions": {
                        "users": acl.users,
                        "groups": acl.groups,
                        "level": acl.level,
                    },
                },
            )
            points.append(point)

        self._batch_upsert(self._child_collection, points)
        logger.info(f"Child {len(children)}개 Upsert 완료")

    def get_parent_by_id(self, parent_id: str) -> Optional[Dict]:
        """Parent ID로 Parent Chunk 원본을 조회합니다."""
        results = self._client.retrieve(
            collection_name=self._parent_collection,
            ids=[parent_id],
            with_payload=True,
        )
        if results:
            return results[0].payload
        return None

    def search_children(
        self,
        dense_vector: List[float],
        sparse_vector: Optional[Tuple[List[int], List[float]]] = None,
        user_id: str = "",
        user_groups: Optional[List[str]] = None,
        top_k: int = 50,
    ) -> List[Dict]:
        """
        Child Chunk를 Hybrid Search(Dense + Sparse)로 검색합니다.
        ACL 필터를 적용하여 권한 있는 문서만 반환합니다.
        """
        # ACL 필터 구성
        acl_conditions = []
        if user_id:
            acl_conditions.append(
                FieldCondition(key="permissions.users", match=MatchAny(any=[user_id]))
            )
        if user_groups:
            acl_conditions.append(
                FieldCondition(key="permissions.groups", match=MatchAny(any=user_groups))
            )
        acl_conditions.append(
            FieldCondition(key="permissions.level", match=MatchAny(any=["public"]))
        )

        search_filter = Filter(should=acl_conditions) if acl_conditions else None

        # Dense Search
        results = self._client.search(
            collection_name=self._child_collection,
            query_vector=NamedVector(name="dense", vector=dense_vector),
            query_filter=search_filter,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results
        ]

    def _batch_upsert(self, collection_name: str, points: List[PointStruct], batch_size: int = 100):
        """배치 단위로 Qdrant에 포인트를 Upsert합니다."""
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(collection_name=collection_name, points=batch)
