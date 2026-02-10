import logging
from dataclasses import dataclass, field
from typing import List, Optional

from fastapi import Header, HTTPException, Request

logger = logging.getLogger(__name__)


@dataclass
class UserContext:
    """인증된 사용자의 컨텍스트 정보."""

    user_id: str
    groups: List[str] = field(default_factory=list)
    email: str = ""


async def extract_user_context(
    request: Request,
    x_user_id: Optional[str] = Header(None),
    x_user_groups: Optional[str] = Header(None),
    x_user_email: Optional[str] = Header(None),
) -> UserContext:
    """
    요청 헤더에서 사용자 컨텍스트를 추출합니다.
    실제 환경에서는 JWT 토큰 검증 후 사용자 정보를 추출합니다.

    Headers:
        X-User-Id: 사용자 ID
        X-User-Groups: 콤마로 구분된 그룹 목록
        X-User-Email: 사용자 이메일
    """
    if not x_user_id:
        raise HTTPException(status_code=401, detail="X-User-Id header is required")

    groups = [g.strip() for g in x_user_groups.split(",") if g.strip()] if x_user_groups else []

    return UserContext(
        user_id=x_user_id,
        groups=groups,
        email=x_user_email or "",
    )
