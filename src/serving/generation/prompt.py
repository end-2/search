from typing import List

from src.serving.retrieval.hybrid import RetrievedDocument

SYSTEM_PROMPT = """당신은 사내 지식 검색 도우미입니다.
아래 제공된 문서(Context)를 기반으로 사용자의 질문에 정확하게 답변해 주세요.

## 규칙
1. 반드시 제공된 Context 내의 정보만을 사용하여 답변하세요. Context에 없는 정보를 추측하거나 생성하지 마세요.
2. 답변에 사용한 문서의 출처(Source)를 반드시 함께 표기하세요.
3. Context에서 답을 찾을 수 없다면 "제공된 문서에서 관련 정보를 찾을 수 없습니다."라고 답변하세요.
4. 답변은 명확하고 구조적으로 작성하세요.

## Context
{context}

## 출처 표기 형식
답변 마지막에 아래 형식으로 출처를 표기하세요:
---
**출처:**
- [문서 제목](URL)
"""


class PromptBuilder:
    """검색된 Context와 사용자 질문을 결합하여 LLM 프롬프트를 구성합니다."""

    def build(
        self, query: str, documents: List[RetrievedDocument]
    ) -> List[dict]:
        """
        LLM에게 전달할 메시지 리스트를 구성합니다.

        Args:
            query: 사용자 질문
            documents: 검색된 문서 리스트 (Parent 문맥 포함)
        Returns:
            OpenAI Chat Completion API 형식의 messages 리스트
        """
        context = self._format_context(documents)
        system_content = SYSTEM_PROMPT.format(context=context)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

    def _format_context(self, documents: List[RetrievedDocument]) -> str:
        """검색된 문서를 Context 문자열로 포맷팅합니다."""
        if not documents:
            return "(검색된 문서가 없습니다)"

        parts = []
        for i, doc in enumerate(documents, 1):
            source_label = f"[{doc.source.upper()}]" if doc.source else ""
            url_label = f"({doc.url})" if doc.url else ""
            author_label = f"by {doc.author}" if doc.author else ""

            header = f"### Document {i} {source_label} {author_label} {url_label}".strip()
            parts.append(f"{header}\n{doc.parent_content}")

        return "\n\n".join(parts)
