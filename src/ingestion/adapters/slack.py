import logging
from datetime import datetime
from typing import Dict, List, Optional

import httpx

from src.common.config import get_settings
from src.ingestion.adapters.base import DataSourceAdapter, DocumentACL, RawDocument

logger = logging.getLogger(__name__)


class SlackAdapter(DataSourceAdapter):
    """Slack 데이터 소스 어댑터. Slack API를 통해 채널 메시지를 수집합니다."""

    BASE_URL = "https://slack.com/api"

    def __init__(self):
        settings = get_settings()
        self._token = settings.slack_bot_token
        self._headers = {"Authorization": f"Bearer {self._token}"}

    @property
    def source_name(self) -> str:
        return "slack"

    def fetch_updates(self, last_sync_time: Optional[datetime] = None) -> List[RawDocument]:
        """Slack 채널에서 메시지를 수집합니다."""
        documents: List[RawDocument] = []
        channels = self._list_channels()

        for channel in channels:
            channel_id = channel["id"]
            channel_name = channel.get("name", "")
            members = self._get_channel_members(channel_id)

            messages = self._fetch_channel_messages(channel_id, last_sync_time)
            for msg in messages:
                if not msg.get("text"):
                    continue

                thread_text = self._fetch_thread_replies(channel_id, msg["ts"])
                full_content = msg["text"]
                if thread_text:
                    full_content += "\n\n[Thread Replies]\n" + thread_text

                doc = RawDocument(
                    doc_id=f"slack_{channel_id}_{msg['ts']}",
                    title=f"#{channel_name}",
                    content=full_content,
                    source="slack",
                    url=f"https://slack.com/archives/{channel_id}/p{msg['ts'].replace('.', '')}",
                    author=msg.get("user", "unknown"),
                    created_at=datetime.fromtimestamp(float(msg["ts"])),
                    metadata={"channel_id": channel_id, "channel_name": channel_name},
                    permissions=self.extract_permissions(
                        {"channel": channel, "members": members}
                    ),
                )
                documents.append(doc)

        logger.info(f"Slack에서 {len(documents)}개 문서를 수집했습니다.")
        return documents

    def extract_permissions(self, raw_data: dict) -> DocumentACL:
        """채널 멤버십 기반으로 ACL을 생성합니다."""
        channel = raw_data.get("channel", {})
        members = raw_data.get("members", [])
        is_private = channel.get("is_private", False)

        return DocumentACL(
            users=members,
            groups=[],
            level="private" if is_private else "public",
        )

    def _list_channels(self) -> List[Dict]:
        """Bot이 참여한 채널 목록을 조회합니다."""
        with httpx.Client() as client:
            resp = client.get(
                f"{self.BASE_URL}/conversations.list",
                headers=self._headers,
                params={"types": "public_channel,private_channel", "limit": 200},
            )
            data = resp.json()
            if not data.get("ok"):
                logger.error(f"Slack conversations.list 실패: {data.get('error')}")
                return []
            return data.get("channels", [])

    def _get_channel_members(self, channel_id: str) -> List[str]:
        """채널의 멤버 ID 목록을 조회합니다."""
        with httpx.Client() as client:
            resp = client.get(
                f"{self.BASE_URL}/conversations.members",
                headers=self._headers,
                params={"channel": channel_id, "limit": 500},
            )
            data = resp.json()
            if not data.get("ok"):
                return []
            return data.get("members", [])

    def _fetch_channel_messages(
        self, channel_id: str, since: Optional[datetime] = None
    ) -> List[Dict]:
        """채널의 메시지를 조회합니다."""
        params: Dict = {"channel": channel_id, "limit": 200}
        if since:
            params["oldest"] = str(since.timestamp())

        with httpx.Client() as client:
            resp = client.get(
                f"{self.BASE_URL}/conversations.history",
                headers=self._headers,
                params=params,
            )
            data = resp.json()
            if not data.get("ok"):
                logger.error(f"Slack history 조회 실패 ({channel_id}): {data.get('error')}")
                return []
            return data.get("messages", [])

    def _fetch_thread_replies(self, channel_id: str, thread_ts: str) -> str:
        """스레드 답글을 조회하여 하나의 문자열로 합칩니다."""
        with httpx.Client() as client:
            resp = client.get(
                f"{self.BASE_URL}/conversations.replies",
                headers=self._headers,
                params={"channel": channel_id, "ts": thread_ts, "limit": 50},
            )
            data = resp.json()
            if not data.get("ok"):
                return ""

            replies = data.get("messages", [])
            # 첫 번째 메시지는 원본이므로 제외
            reply_texts = [r.get("text", "") for r in replies[1:] if r.get("text")]
            return "\n".join(reply_texts)
