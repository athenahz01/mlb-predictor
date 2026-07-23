from __future__ import annotations

import pytest
from fastapi import HTTPException

from athena_api.auth import current_user
from athena_api.settings import Settings


def test_local_auth_fallback_is_explicit():
    user = current_user(None, Settings(auth_required=False))
    assert user.id == "local-development"


def test_auth_required_rejects_missing_session():
    with pytest.raises(HTTPException) as caught:
        current_user(None, Settings(auth_required=True))
    assert caught.value.status_code == 401
