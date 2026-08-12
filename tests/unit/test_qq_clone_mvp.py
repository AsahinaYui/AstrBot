"""Focused tests for the small QQClone learning and shadow-draft MVP."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from astrbot.builtin_stars.qq_clone_observer._conversation import (
    _rebuild_conversation_for_normalized,
    _update_conversation_privacy,
)
from astrbot.builtin_stars.qq_clone_observer._mvp import (
    _capture_human_example,
    _get_persona_profile,
    _prepare_shadow_request,
    _review_shadow_draft,
    _segments_to_text,
    _store_shadow_draft,
)
from astrbot.builtin_stars.qq_clone_observer.main import QQCloneObserver, _build_schema
from astrbot.core.provider.entities import LLMResponse


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    """Create a complete QQClone SQLite schema for an MVP test.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Open SQLite connection.
    """
    connection = sqlite3.connect(tmp_path / "qqclone-mvp.db", check_same_thread=False)
    connection.execute("PRAGMA foreign_keys=ON")
    _build_schema(connection.cursor())
    return connection


def _insert_event(
    conn: sqlite3.Connection,
    *,
    account_role: str,
    direction: str,
    actor_provenance: str,
    text: str,
    platform_id: str,
    self_id: str,
    conversation_id: str,
    sender_id: str,
    message_id: str,
) -> int:
    """Insert one immutable raw/normalized fact and materialize its conversation.

    Args:
        conn: SQLite connection.
        account_role: Source or clone account role.
        direction: Incoming or outgoing direction.
        actor_provenance: Human, agent, or unknown actor classification.
        text: Plain message text.
        platform_id: Configured platform ID.
        self_id: Configured QQ self ID.
        conversation_id: Stable normalized conversation ID.
        sender_id: Sender or private peer ID.
        message_id: Stable source message ID.

    Returns:
        Inserted normalized event ID.
    """
    payload = {
        "message_type": "private",
        "user_id": sender_id,
        "message": [{"type": "text", "data": {"text": text}}],
    }
    raw = conn.execute(
        "INSERT INTO raw_events (platform_id, self_id, account_role, "
        "source_message_id, message_direction, provenance, payload_json, "
        "payload_sha256, stable_fingerprint, idempotency_key, parser_version, "
        "received_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            platform_id,
            self_id,
            account_role,
            message_id,
            direction,
            actor_provenance,
            json.dumps(payload, ensure_ascii=False),
            f"sha-{message_id}",
            f"fingerprint-{message_id}",
            f"message:{platform_id}:{self_id}:{message_id}",
            "test",
            "1000",
        ),
    )
    normalized = conn.execute(
        "INSERT INTO normalized_events (raw_event_id, parser_version, platform_id, "
        "self_id, account_role, source_message_id, conversation_type, "
        "conversation_id, sender_id, direction, provenance, actor_provenance, "
        "normalized_segments_json, normalized_processing_time) "
        "VALUES (?, ?, ?, ?, ?, ?, 'private', ?, ?, ?, ?, ?, ?, ?)",
        (
            raw.lastrowid,
            "test",
            platform_id,
            self_id,
            account_role,
            message_id,
            conversation_id,
            sender_id,
            direction,
            actor_provenance,
            actor_provenance,
            json.dumps([{"kind": "text", "data": {"text": text}}]),
            "1000",
        ),
    )
    normalized_id = int(normalized.lastrowid)
    conn.commit()
    _rebuild_conversation_for_normalized(conn, normalized_id, 12000)
    return normalized_id


def test_segments_to_text_redacts_common_contact_details():
    segments = json.dumps([{"kind": "text", "data": {"text": "a@b.com 13812345678"}}])
    assert _segments_to_text(segments) == "[邮箱] [手机号]"


@pytest.mark.parametrize(
    ("role", "direction", "actor"),
    [
        ("source", "incoming", "unknown"),
        ("source", "outgoing", "agent"),
        ("clone", "outgoing", "human"),
    ],
)
def test_only_verified_source_human_outgoing_is_eligible(
    conn: sqlite3.Connection,
    role: str,
    direction: str,
    actor: str,
):
    event_id = _insert_event(
        conn,
        account_role=role,
        direction=direction,
        actor_provenance=actor,
        text="不应该学习",
        platform_id="source" if role == "source" else "clone",
        self_id="100" if role == "source" else "200",
        conversation_id=f"{role}:private:300",
        sender_id="300",
        message_id=f"{role}-{direction}-{actor}",
    )
    assert not _capture_human_example(conn, event_id)
    assert conn.execute("SELECT COUNT(*) FROM human_examples").fetchone()[0] == 0


def test_human_example_builds_persona_and_honors_later_exclusion(
    conn: sqlite3.Connection,
):
    event_id = _insert_event(
        conn,
        account_role="source",
        direction="outgoing",
        actor_provenance="human",
        text="好呀，晚点聊？",
        platform_id="source",
        self_id="100",
        conversation_id="source:private:300",
        sender_id="300",
        message_id="human-1",
    )
    assert _capture_human_example(conn, event_id)
    profile = _get_persona_profile(conn, "source")
    assert profile is not None and profile["sample_count"] == 1

    _update_conversation_privacy(
        conn,
        "source:private:300",
        privacy_level="excluded",
    )
    assert _get_persona_profile(conn, "source") is None


def test_shadow_request_uses_verified_examples_and_skips_admin_command(
    conn: sqlite3.Connection,
):
    source_event = _insert_event(
        conn,
        account_role="source",
        direction="outgoing",
        actor_provenance="human",
        text="可以呀",
        platform_id="source",
        self_id="100",
        conversation_id="source:private:300",
        sender_id="300",
        message_id="human-2",
    )
    _capture_human_example(conn, source_event)
    clone_event = _insert_event(
        conn,
        account_role="clone",
        direction="incoming",
        actor_provenance="unknown",
        text="今晚有空吗",
        platform_id="clone",
        self_id="200",
        conversation_id="clone:private:300",
        sender_id="300",
        message_id="clone-1",
    )
    request = _prepare_shadow_request(conn, clone_event, "source", "100")
    assert request is not None
    assert request["incoming_text"] == "今晚有空吗"
    assert [example["response"] for example in request["examples"]] == ["可以呀"]

    command_event = _insert_event(
        conn,
        account_role="clone",
        direction="incoming",
        actor_provenance="unknown",
        text="/qqclone_drafts",
        platform_id="clone",
        self_id="200",
        conversation_id="clone:private:300",
        sender_id="300",
        message_id="clone-command",
    )
    assert _prepare_shadow_request(conn, command_event, "source", "100") is None


def test_shadow_storage_is_idempotent_and_review_never_creates_an_action(
    conn: sqlite3.Connection,
):
    event_id = _insert_event(
        conn,
        account_role="clone",
        direction="incoming",
        actor_provenance="unknown",
        text="在吗",
        platform_id="clone",
        self_id="200",
        conversation_id="clone:private:300",
        sender_id="300",
        message_id="clone-2",
    )
    request = _prepare_shadow_request(conn, event_id, "source", "100")
    assert request is not None
    first = _store_shadow_draft(
        conn, request, "source", "100", "provider", "model", "在呀"
    )
    second = _store_shadow_draft(
        conn, request, "source", "100", "provider", "model", "另一个结果"
    )
    assert first == second
    reviewed = _review_shadow_draft(conn, first, "edited", "刚看到，在呀")
    assert reviewed is not None and reviewed["edited_text"] == "刚看到，在呀"
    assert conn.execute("SELECT COUNT(*) FROM shadow_drafts").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_plugin_generates_and_persists_without_sending(conn: sqlite3.Connection):
    event_id = _insert_event(
        conn,
        account_role="clone",
        direction="incoming",
        actor_provenance="unknown",
        text="吃饭了吗",
        platform_id="clone",
        self_id="200",
        conversation_id="clone:private:300",
        sender_id="300",
        message_id="clone-3",
    )
    provider = MagicMock()
    provider.meta.return_value = MagicMock(id="provider", model="model")
    provider.text_chat = AsyncMock(return_value=LLMResponse("assistant", "吃啦"))
    context = MagicMock()
    context.get_using_provider.return_value = provider
    config = {
        "shadow_enabled": True,
        "accounts": [
            {
                "platform_id": "source",
                "self_id": "100",
                "role": "source",
                "enabled": True,
            },
            {
                "platform_id": "clone",
                "self_id": "200",
                "role": "clone",
                "enabled": True,
            },
        ],
    }
    plugin = QQCloneObserver(context, config)
    plugin._conn = conn
    await plugin._generate_shadow_draft(event_id, "clone:FriendMessage:300")

    row = conn.execute(
        "SELECT draft_text, status FROM shadow_drafts WHERE normalized_event_id = ?",
        (event_id,),
    ).fetchone()
    assert row == ("吃啦", "pending")
    context.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_provider_failure_is_stored_as_reviewable_failure(
    conn: sqlite3.Connection,
):
    event_id = _insert_event(
        conn,
        account_role="clone",
        direction="incoming",
        actor_provenance="unknown",
        text="你好",
        platform_id="clone",
        self_id="200",
        conversation_id="clone:private:400",
        sender_id="400",
        message_id="clone-4",
    )
    provider = MagicMock()
    provider.meta.return_value = MagicMock(id="provider", model="model")
    provider.text_chat = AsyncMock(side_effect=RuntimeError("secret response"))
    context = MagicMock()
    context.get_using_provider.return_value = provider
    plugin = QQCloneObserver(
        context,
        {
            "shadow_enabled": True,
            "accounts": [
                {
                    "platform_id": "source",
                    "self_id": "100",
                    "role": "source",
                    "enabled": True,
                }
            ],
        },
    )
    plugin._conn = conn
    await plugin._generate_shadow_draft(event_id, "clone:FriendMessage:400")
    row = conn.execute(
        "SELECT status, error_type FROM shadow_drafts WHERE normalized_event_id = ?",
        (event_id,),
    ).fetchone()
    assert row == ("failed", "RuntimeError")
