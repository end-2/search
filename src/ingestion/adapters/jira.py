import logging
from datetime import datetime
from typing import Dict, List, Optional

import httpx

from src.common.config import get_settings
from src.ingestion.adapters.base import DataSourceAdapter, DocumentACL, RawDocument

logger = logging.getLogger(__name__)


class JiraAdapter(DataSourceAdapter):
    """Jira 데이터 소스 어댑터. Jira REST API를 통해 이슈를 수집합니다."""

    def __init__(self):
        settings = get_settings()
        self._base_url = settings.jira_base_url.rstrip("/")
        self._auth = (settings.jira_email, settings.jira_api_token)

    @property
    def source_name(self) -> str:
        return "jira"

    def fetch_updates(self, last_sync_time: Optional[datetime] = None) -> List[RawDocument]:
        """Jira에서 이슈를 수집합니다."""
        jql = "ORDER BY updated DESC"
        if last_sync_time:
            formatted = last_sync_time.strftime("%Y-%m-%d %H:%M")
            jql = f'updated >= "{formatted}" ORDER BY updated DESC'

        documents: List[RawDocument] = []
        start_at = 0
        max_results = 50

        while True:
            issues = self._search_issues(jql, start_at, max_results)
            if not issues:
                break

            for issue in issues:
                doc = self._issue_to_document(issue)
                documents.append(doc)

            if len(issues) < max_results:
                break
            start_at += max_results

        logger.info(f"Jira에서 {len(documents)}개 이슈를 수집했습니다.")
        return documents

    def extract_permissions(self, raw_data: dict) -> DocumentACL:
        """Jira 프로젝트 기반으로 ACL을 생성합니다."""
        project_key = raw_data.get("project_key", "")
        assignee = raw_data.get("assignee", "")
        reporter = raw_data.get("reporter", "")

        users = [u for u in [assignee, reporter] if u]
        groups = [f"jira-{project_key.lower()}"] if project_key else []

        return DocumentACL(
            users=users,
            groups=groups,
            level="private",
        )

    def _search_issues(self, jql: str, start_at: int, max_results: int) -> List[Dict]:
        """JQL로 이슈를 검색합니다."""
        with httpx.Client() as client:
            resp = client.get(
                f"{self._base_url}/rest/api/3/search",
                auth=self._auth,
                params={
                    "jql": jql,
                    "startAt": start_at,
                    "maxResults": max_results,
                    "fields": "summary,description,status,assignee,reporter,project,created,updated,comment",
                },
            )
            if resp.status_code != 200:
                logger.error(f"Jira 검색 실패: {resp.status_code} {resp.text}")
                return []
            return resp.json().get("issues", [])

    def _issue_to_document(self, issue: Dict) -> RawDocument:
        """Jira 이슈를 RawDocument로 변환합니다."""
        fields = issue.get("fields", {})
        key = issue.get("key", "")

        summary = fields.get("summary", "")
        description = self._extract_text(fields.get("description"))
        comments = self._extract_comments(fields.get("comment", {}))

        content_parts = [f"[{key}] {summary}"]
        if description:
            content_parts.append(f"\n{description}")
        if comments:
            content_parts.append(f"\n[Comments]\n{comments}")

        project = fields.get("project", {})
        assignee = fields.get("assignee") or {}
        reporter = fields.get("reporter") or {}
        status = fields.get("status", {})

        created_str = fields.get("created", "")
        created_at = None
        if created_str:
            created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))

        return RawDocument(
            doc_id=f"jira_{key}",
            title=f"[{key}] {summary}",
            content="\n".join(content_parts),
            source="jira",
            url=f"{self._base_url}/browse/{key}",
            author=reporter.get("displayName", ""),
            created_at=created_at,
            metadata={
                "project_key": project.get("key", ""),
                "status": status.get("name", ""),
                "assignee": assignee.get("displayName", ""),
            },
            permissions=self.extract_permissions(
                {
                    "project_key": project.get("key", ""),
                    "assignee": assignee.get("emailAddress", ""),
                    "reporter": reporter.get("emailAddress", ""),
                }
            ),
        )

    def _extract_text(self, adf_content) -> str:
        """Atlassian Document Format(ADF) JSON을 평문으로 변환합니다."""
        if not adf_content:
            return ""
        if isinstance(adf_content, str):
            return adf_content

        texts = []
        self._walk_adf(adf_content, texts)
        return " ".join(texts)

    def _walk_adf(self, node: dict, texts: list):
        """ADF 트리를 재귀 탐색하여 텍스트를 추출합니다."""
        if node.get("type") == "text":
            texts.append(node.get("text", ""))
        for child in node.get("content", []):
            if isinstance(child, dict):
                self._walk_adf(child, texts)

    def _extract_comments(self, comment_data: dict) -> str:
        """이슈의 코멘트를 하나의 문자열로 합칩니다."""
        comments = comment_data.get("comments", [])
        parts = []
        for c in comments:
            author = c.get("author", {}).get("displayName", "unknown")
            body = self._extract_text(c.get("body"))
            if body:
                parts.append(f"- {author}: {body}")
        return "\n".join(parts)
