"""Tests for QQClone passive observer plugin.

Verifies:
- Safe JSON serialization of various payloads.
- Canonical SHA-256 fingerprint determinism.
- Identity account resolution (fail-closed).
- In-memory health state tracking.
- SQLite schema and immutability triggers.
- Idempotency priority ordering.
- No outbound message sending.
"""

import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from astrbot.builtin_stars.qq_clone_observer._normalized import (
    _build_conversation_id,
    _build_normalized_schema,
    _derive_actor_provenance,
    _derive_conversation_type,
    _insert_normalized_event,
    _normalize_event,
    _normalize_raw_segments,
    _normalize_segment,
    _normalize_segments,
)
from astrbot.builtin_stars.qq_clone_observer.main import (
    QQCloneObserver,
    _build_schema,
    _canonical_fingerprint,
    _check_existing_by_event_id,
    _check_existing_by_fingerprint,
    _check_existing_by_platform_and_msg_id,
    _insert_raw_event,
    _safe_json_serialize,
)
from astrbot.core.platform.astrbot_message import (
    AstrBotMessage,
    Group,
    MessageDirection,
    MessageMember,
    MessageType,
)


class TestSafeJsonSerialize:
    """Tests for _safe_json_serialize."""

    def test_primitive_string(self):
        assert json.loads(_safe_json_serialize("hello")) == "hello"

    def test_primitive_int(self):
        assert json.loads(_safe_json_serialize(42)) == 42

    def test_primitive_float(self):
        assert json.loads(_safe_json_serialize(3.14)) == 3.14

    def test_primitive_bool(self):
        assert json.loads(_safe_json_serialize(True)) is True

    def test_primitive_none(self):
        assert json.loads(_safe_json_serialize(None)) is None

    def test_bytes_converts_to_hex(self):
        result = json.loads(_safe_json_serialize(b"\x00\x01"))
        assert isinstance(result, str)
        assert result == "0001"

    def test_flat_dict(self):
        data = {"key": "value", "num": 1}
        assert json.loads(_safe_json_serialize(data)) == data

    def test_nested_dict(self):
        data = {"outer": {"inner": "val", "count": 2}}
        assert json.loads(_safe_json_serialize(data)) == data

    def test_list(self):
        data = [1, "two", 3.0]
        assert json.loads(_safe_json_serialize(data)) == data

    def test_list_of_dicts(self):
        data = [{"a": 1}, {"b": 2}]
        assert json.loads(_safe_json_serialize(data)) == data

    def test_unsupported_object_falls_back_to_str(self):
        class Unsupported:
            def __str__(self) -> str:
                return "fallback_repr"

        obj = Unsupported()
        result = json.loads(_safe_json_serialize(obj))
        assert result == "fallback_repr"

    def test_max_depth_prevents_infinite_recursion(self):
        inner = {}
        outer = {"key": inner}
        inner["parent"] = outer
        result = json.loads(_safe_json_serialize(outer, max_depth=2))
        assert isinstance(result, dict)
        assert "key" in result

    def test_onebot_like_payload(self):
        payload = {
            "post_type": "message",
            "message_type": "group",
            "self_id": 123456,
            "user_id": 789012,
            "message": [
                {"type": "text", "data": {"text": "hello"}},
                {"type": "at", "data": {"qq": 123456}},
            ],
        }
        result = json.loads(_safe_json_serialize(payload))
        assert result["post_type"] == "message"
        assert result["self_id"] == 123456
        assert len(result["message"]) == 2
        assert result["message"][0]["data"]["text"] == "hello"


class TestCanonicalFingerprint:
    """Tests for _canonical_fingerprint."""

    def test_deterministic(self):
        fp1 = _canonical_fingerprint("p1", "bot1", "incoming", "qq", "sha1")
        fp2 = _canonical_fingerprint("p1", "bot1", "incoming", "qq", "sha1")
        assert fp1 == fp2

    def test_different_inputs_different_fingerprints(self):
        fp1 = _canonical_fingerprint("p1", "bot1", "incoming", "qq", "sha1")
        fp2 = _canonical_fingerprint("p1", "bot1", "incoming", "qq", "sha2")
        assert fp1 != fp2

    def test_none_provenance(self):
        fp = _canonical_fingerprint("p1", "bot1", "incoming", None, "sha1")
        assert isinstance(fp, str)
        assert len(fp) == 64

    def test_all_empty_fields(self):
        fp = _canonical_fingerprint("", "", None, None, "")
        assert isinstance(fp, str)
        assert len(fp) == 64


class TestSQLiteSchema:
    """Tests for SQLite schema creation and operations."""

    @pytest.fixture
    def conn(self, tmp_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(str(tmp_path / "test_raw_events.db"))
        conn.execute("PRAGMA journal_mode=WAL")
        _build_schema(conn.cursor())
        return conn

    def test_table_exists(self, conn: sqlite3.Connection):
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='raw_events'",
        )
        assert cursor.fetchone() is not None

    def test_columns_exist(self, conn: sqlite3.Connection):
        cursor = conn.execute("PRAGMA table_info(raw_events)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            "id",
            "platform_id",
            "self_id",
            "account_role",
            "source_event_type",
            "source_event_id",
            "source_message_id",
            "message_direction",
            "provenance",
            "payload_json",
            "payload_sha256",
            "stable_fingerprint",
            "idempotency_key",
            "parser_version",
            "received_time",
            "processing_status",
        }
        assert expected.issubset(columns)

    def test_insert_and_select(self, conn: sqlite3.Connection):
        _insert_raw_event(
            conn,
            "test_platform",
            "bot1",
            "source",
            "message",
            None,
            "msg_001",
            "incoming",
            "qq",
            '{"text":"hello"}',
            "abc123",
            "fp123",
            "message:test_platform:bot1:msg_001",
        )
        cursor = conn.execute("SELECT * FROM raw_events")
        row = cursor.fetchone()
        assert row is not None
        assert row[1] == "test_platform"
        assert row[2] == "bot1"
        assert row[3] == "source"
        assert row[4] == "message"
        assert row[6] == "msg_001"
        assert row[7] == "incoming"
        assert row[8] == "qq"
        assert row[9] == '{"text":"hello"}'
        assert row[10] == "abc123"
        assert row[13] == "0.1.0"
        assert row[15] == "accepted"

    def test_update_trigger_rejects_update(self, conn: sqlite3.Connection):
        _insert_raw_event(
            conn,
            "p",
            "b",
            "source",
            "msg",
            None,
            "m1",
            "in",
            "qq",
            "{}",
            "sha",
            "fp",
            "message:p:b:m1",
        )
        with pytest.raises(sqlite3.DatabaseError, match="UPDATE"):
            conn.execute("UPDATE raw_events SET processing_status = 'rejected'")

    def test_delete_trigger_rejects_delete(self, conn: sqlite3.Connection):
        _insert_raw_event(
            conn,
            "p",
            "b",
            "source",
            "msg",
            None,
            "m1",
            "in",
            "qq",
            "{}",
            "sha",
            "fp",
            "message:p:b:m1",
        )
        with pytest.raises(sqlite3.DatabaseError, match="DELETE"):
            conn.execute("DELETE FROM raw_events")


class TestIdempotencyChecks:
    """Tests for idempotency check functions."""

    @pytest.fixture
    def conn(self, tmp_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(str(tmp_path / "test_idem.db"))
        conn.execute("PRAGMA journal_mode=WAL")
        _build_schema(conn.cursor())
        _insert_raw_event(
            conn,
            "p1",
            "bot1",
            "source",
            "message",
            None,
            "msg_001",
            "incoming",
            "qq",
            '{"t":"hello"}',
            "sha_001",
            "fp_001",
            "message:p1:bot1:msg_001",
        )
        return conn

    def test_insert_reports_database_duplicate(self, conn: sqlite3.Connection):
        assert (
            _insert_raw_event(
                conn,
                "p1",
                "bot1",
                "source",
                "message",
                None,
                "msg_001",
                "incoming",
                "qq",
                '{"t":"hello"}',
                "sha_001",
                "fp_001",
                "message:p1:bot1:msg_001",
            )
            is None
        )

    def test_check_by_event_id_not_found(self, conn: sqlite3.Connection):
        assert _check_existing_by_event_id(conn, "p1", "bot1", "evt_001") is False

    def test_check_by_event_id_scoped_to_account(self, conn: sqlite3.Connection):
        assert _check_existing_by_event_id(conn, "p1", "other_bot", "evt_001") is False

    def test_check_by_platform_and_msg_id_found(self, conn: sqlite3.Connection):
        assert (
            _check_existing_by_platform_and_msg_id(conn, "p1", "bot1", "msg_001")
            is True
        )

    def test_check_by_platform_and_msg_id_not_found(self, conn: sqlite3.Connection):
        assert (
            _check_existing_by_platform_and_msg_id(conn, "p1", "bot1", "other") is False
        )

    def test_check_by_platform_and_msg_id_wrong_platform(
        self,
        conn: sqlite3.Connection,
    ):
        assert (
            _check_existing_by_platform_and_msg_id(conn, "p2", "bot1", "msg_001")
            is False
        )

    def test_check_by_platform_and_msg_id_wrong_self_id(
        self,
        conn: sqlite3.Connection,
    ):
        assert (
            _check_existing_by_platform_and_msg_id(conn, "p1", "other_bot", "msg_001")
            is False
        )

    def test_check_by_fingerprint_found(self, conn: sqlite3.Connection):
        assert _check_existing_by_fingerprint(conn, "fp_001") is True

    def test_check_by_fingerprint_not_found(self, conn: sqlite3.Connection):
        assert _check_existing_by_fingerprint(conn, "fp_none") is False


class TestAccountResolution:
    """Tests for _resolve_account_role fail-closed behavior."""

    @pytest.fixture
    def plugin(self):
        ctx = MagicMock()
        config = {
            "accounts": [
                {
                    "platform_id": "aiocqhttp_source",
                    "self_id": "123456",
                    "role": "source",
                    "enabled": True,
                },
                {
                    "platform_id": "aiocqhttp_clone",
                    "self_id": "789012",
                    "role": "clone",
                    "enabled": True,
                },
                {
                    "platform_id": "aiocqhttp_disabled",
                    "self_id": "111111",
                    "role": "source",
                    "enabled": False,
                },
                {
                    "platform_id": "aiocqhttp_bad_role",
                    "self_id": "222222",
                    "role": "invalid_role",
                    "enabled": True,
                },
            ],
        }
        config_mock = MagicMock()
        config_mock.get.side_effect = lambda key, default=None: config.get(key, default)
        plugin = QQCloneObserver(ctx, config_mock)
        return plugin

    def test_unknown_platform_rejected(self, plugin):
        role, expected, reason = plugin._resolve_account_role(
            "unknown_platform",
            "bot1",
        )
        assert role is None
        assert reason == "unknown platform_id"

    def test_disabled_account_rejected(self, plugin):
        role, expected, reason = plugin._resolve_account_role(
            "aiocqhttp_disabled",
            "111111",
        )
        assert role is None
        assert reason == "account disabled"

    def test_self_id_mismatch_rejected(self, plugin):
        role, expected, reason = plugin._resolve_account_role(
            "aiocqhttp_source",
            "wrong_id",
        )
        assert role is None
        assert "self_id mismatch" in reason

    def test_invalid_role_rejected(self, plugin):
        role, expected, reason = plugin._resolve_account_role(
            "aiocqhttp_bad_role",
            "222222",
        )
        assert role is None
        assert "invalid role" in reason

    def test_duplicate_platform_id_rejected(self, plugin):
        # Add a duplicate mapping to the config
        plugin._config.get.side_effect = lambda key, default=None: {
            "accounts": [
                {
                    "platform_id": "dup_platform",
                    "self_id": "123",
                    "role": "source",
                    "enabled": True,
                },
                {
                    "platform_id": "dup_platform",
                    "self_id": "456",
                    "role": "clone",
                    "enabled": True,
                },
            ],
        }.get(key, default)
        role, expected, reason = plugin._resolve_account_role(
            "dup_platform",
            "123",
        )
        assert role is None
        assert "duplicate" in reason

    def test_malformed_mapping_is_rejected_safely(self, plugin):
        plugin._config.get.side_effect = lambda key, default=None: {
            "accounts": ["invalid", None]
        }.get(key, default)
        role, expected, reason = plugin._resolve_account_role("p1", "bot1")
        assert role is None
        assert expected is None
        assert reason == "unknown platform_id"

    def test_valid_source_accepted(self, plugin):
        role, expected, reason = plugin._resolve_account_role(
            "aiocqhttp_source",
            "123456",
        )
        assert role == "source"
        assert reason is None

    def test_valid_clone_accepted(self, plugin):
        role, expected, reason = plugin._resolve_account_role(
            "aiocqhttp_clone",
            "789012",
        )
        assert role == "clone"
        assert reason is None


class TestHealthTracking:
    """Tests for in-memory health state tracking."""

    @pytest.fixture
    def plugin(self):
        ctx = MagicMock()
        config = {
            "accounts": [
                {
                    "platform_id": "aiocqhttp_source",
                    "self_id": "123456",
                    "role": "source",
                    "enabled": True,
                },
            ],
        }
        config_mock = MagicMock()
        config_mock.get.side_effect = lambda key, default=None: config.get(key, default)
        plugin = QQCloneObserver(ctx, config_mock)
        plugin._rebuild_health()
        return plugin

    def test_health_initialized_from_config(self, plugin):
        health = plugin.get_health()
        assert "aiocqhttp_source" in health
        assert health["aiocqhttp_source"]["configured_role"] == "source"
        assert health["aiocqhttp_source"]["expected_self_id"] == "123456"
        assert health["aiocqhttp_source"]["accepted_count"] == 0
        assert health["aiocqhttp_source"]["rejected_count"] == 0
        assert health["aiocqhttp_source"]["duplicate_count"] == 0

    def test_health_is_read_only_copy(self, plugin):
        health = plugin.get_health()
        health["aiocqhttp_source"]["accepted_count"] = 999
        actual = plugin.get_health()
        assert actual["aiocqhttp_source"]["accepted_count"] == 0

    @pytest.mark.asyncio
    async def test_health_tracks_unknown_platform(self, plugin):
        """Unknown platform events should be tracked in health state."""
        event = MagicMock()
        event.get_platform_id.return_value = "unknown_platform"
        event.get_self_id.return_value = "bot1"
        event.get_sender_id.return_value = "user_001"
        event.send = MagicMock()
        event.send_streaming = MagicMock()
        event.plain_result = MagicMock()
        event.make_result = MagicMock()
        message_obj = MagicMock()
        message_obj.direction = MessageDirection.UNKNOWN
        message_obj.provenance = "unknown"
        message_obj.message_id = "msg_001"
        message_obj.timestamp = 1000
        message_obj.raw_message = None
        message_obj.self_id = "bot1"
        message_obj.sender = MessageMember(user_id="user_001")
        event.message_obj = message_obj
        await plugin.on_message_observed(event)
        health = plugin.get_health()
        assert "unknown_platform" in health
        assert health["unknown_platform"]["configured_role"] == "unknown"

    @pytest.fixture
    def plugin_with_db(self, tmp_path: Path):
        ctx = MagicMock()
        config = {
            "accounts": [
                {
                    "platform_id": "aiocqhttp_source",
                    "self_id": "123456",
                    "role": "source",
                    "enabled": True,
                },
            ],
        }
        config_mock = MagicMock()
        config_mock.get.side_effect = lambda key, default=None: config.get(key, default)
        with patch(
            "astrbot.builtin_stars.qq_clone_observer.main.get_astrbot_plugin_data_path",
            return_value=str(tmp_path),
        ):
            plugin_with_db = QQCloneObserver(ctx, config_mock)
            plugin_with_db._conn = sqlite3.connect(
                str(tmp_path / "raw_events.db"), check_same_thread=False
            )
            plugin_with_db._conn.execute("PRAGMA journal_mode=WAL")
            _build_schema(plugin_with_db._conn.cursor())
            plugin_with_db._rebuild_health()
            return plugin_with_db

    def _make_event(
        self,
        platform_id: str = "aiocqhttp_source",
        self_id: str = "123456",
        message_id: str = "msg_001",
        direction: MessageDirection = MessageDirection.INCOMING,
        provenance: str = "qq",
        sender_id: str = "user_001",
        raw_message: object = None,
    ) -> MagicMock:
        event = MagicMock()
        event.get_platform_id.return_value = platform_id
        event.get_self_id.return_value = self_id
        event.get_sender_id.return_value = sender_id

        message_obj = MagicMock(spec=AstrBotMessage)
        message_obj.direction = direction
        message_obj.provenance = provenance
        message_obj.message_id = message_id
        message_obj.timestamp = int(time.time())
        message_obj.raw_message = raw_message
        message_obj.self_id = self_id
        message_obj.sender = MessageMember(user_id=sender_id)

        event.message_obj = message_obj
        return event

    @pytest.mark.asyncio
    async def test_valid_incoming_event_stored(self, plugin_with_db):
        raw = {"post_type": "message", "message_type": "group", "self_id": 123456}
        event = self._make_event(raw_message=raw)
        await plugin_with_db.on_message_observed(event)

        cursor = plugin_with_db._conn.execute("SELECT * FROM raw_events")
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][1] == "aiocqhttp_source"
        assert rows[0][2] == "123456"
        assert rows[0][3] == "source"
        assert rows[0][4] == "message"
        assert rows[0][6] == "msg_001"
        assert rows[0][7] == "incoming"
        assert rows[0][8] == "qq"

        health = plugin_with_db.get_health()
        assert health["aiocqhttp_source"]["accepted_count"] == 1
        assert health["aiocqhttp_source"]["rejected_count"] == 0
        assert health["aiocqhttp_source"]["duplicate_count"] == 0

    @pytest.mark.asyncio
    async def test_duplicate_message_id_skipped(self, plugin_with_db):
        raw = {"post_type": "message"}
        event = self._make_event(raw_message=raw)
        await plugin_with_db.on_message_observed(event)
        await plugin_with_db.on_message_observed(event)

        cursor = plugin_with_db._conn.execute("SELECT * FROM raw_events")
        rows = cursor.fetchall()
        assert len(rows) == 1

        health = plugin_with_db.get_health()
        assert health["aiocqhttp_source"]["accepted_count"] == 1
        assert health["aiocqhttp_source"]["duplicate_count"] == 1

    @pytest.mark.asyncio
    async def test_unknown_platform_rejected(self, plugin_with_db):
        event = self._make_event(platform_id="unknown_platform", self_id="bot1")
        await plugin_with_db.on_message_observed(event)

        cursor = plugin_with_db._conn.execute("SELECT * FROM raw_events")
        rows = cursor.fetchall()
        assert len(rows) == 0

        health = plugin_with_db.get_health()
        assert health["unknown_platform"]["rejected_count"] == 1

    @pytest.mark.asyncio
    async def test_self_id_mismatch_rejected(self, plugin_with_db):
        event = self._make_event(self_id="wrong_id")
        await plugin_with_db.on_message_observed(event)

        cursor = plugin_with_db._conn.execute("SELECT * FROM raw_events")
        rows = cursor.fetchall()
        assert len(rows) == 0

    @pytest.mark.asyncio
    async def test_outgoing_message_stored_with_direction(self, plugin_with_db):
        raw = {"post_type": "message_sent"}
        event = self._make_event(raw_message=raw, direction=MessageDirection.OUTGOING)
        await plugin_with_db.on_message_observed(event)

        cursor = plugin_with_db._conn.execute("SELECT * FROM raw_events")
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][7] == "outgoing"

    @pytest.mark.asyncio
    async def test_plugin_never_sends(self, plugin_with_db):
        event = self._make_event(raw_message={"post_type": "message"})
        await plugin_with_db.on_message_observed(event)

        event.send.assert_not_called()
        event.send_streaming.assert_not_called()
        event.plain_result.assert_not_called()
        event.make_result.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_raw_message_serializes_safely(self, plugin_with_db):
        event = self._make_event(raw_message=None)
        await plugin_with_db.on_message_observed(event)

        cursor = plugin_with_db._conn.execute("SELECT * FROM raw_events")
        rows = cursor.fetchall()
        assert len(rows) == 1
        payload = json.loads(rows[0][9])
        assert payload is None

    @pytest.mark.asyncio
    async def test_remaining_health_state_after_rejection_then_acceptance(
        self, plugin_with_db
    ):
        event1 = self._make_event(self_id="wrong_id")
        await plugin_with_db.on_message_observed(event1)

        event2 = self._make_event(raw_message={"post_type": "message"})
        await plugin_with_db.on_message_observed(event2)

        health = plugin_with_db.get_health()
        assert health["aiocqhttp_source"]["accepted_count"] == 1
        assert health["aiocqhttp_source"]["rejected_count"] == 1
        assert health["aiocqhttp_source"]["duplicate_count"] == 0

    @pytest.mark.asyncio
    async def test_no_db_conn_does_not_crash(self, plugin_with_db):
        plugin_with_db._conn = None
        event = self._make_event(raw_message={"post_type": "message"})
        await plugin_with_db.on_message_observed(event)

    @pytest.mark.asyncio
    async def test_config_not_list_handles_gracefully(self, plugin_with_db):
        config_mock = MagicMock()
        config_mock.get.side_effect = lambda key, default=None: (
            "not_a_list" if key == "accounts" else default
        )
        plugin_with_db._config = config_mock
        event = self._make_event(raw_message={"post_type": "message"})
        await plugin_with_db.on_message_observed(event)
        cursor = plugin_with_db._conn.execute("SELECT * FROM raw_events")
        assert len(cursor.fetchall()) == 0

    @pytest.mark.asyncio
    async def test_aiocqhttp_event_object_serializes(self, plugin_with_db):
        class FakeAiocqhttpEvent:
            def __init__(self) -> None:
                self.post_type = "message"
                self.message_type = "group"
                self.self_id = 123456

            def get(self, key: str, default=None):
                return getattr(self, key, default)

        raw = FakeAiocqhttpEvent()
        event = self._make_event(raw_message=raw)
        await plugin_with_db.on_message_observed(event)

        cursor = plugin_with_db._conn.execute("SELECT * FROM raw_events")
        rows = cursor.fetchall()
        assert len(rows) == 1
        payload = json.loads(rows[0][9])
        assert isinstance(payload, str)
        assert "FakeAiocqhttpEvent" in payload
        assert rows[0][4] == "message"

    @pytest.mark.asyncio
    async def test_sha256_fallback_idempotency(self, plugin_with_db):
        raw = {"post_type": "message"}
        event = self._make_event(raw_message=raw, message_id="")
        await plugin_with_db.on_message_observed(event)
        event2 = self._make_event(raw_message=raw, message_id="")
        await plugin_with_db.on_message_observed(event2)

        cursor = plugin_with_db._conn.execute("SELECT * FROM raw_events")
        rows = cursor.fetchall()
        assert len(rows) == 1

        health = plugin_with_db.get_health()
        assert health["aiocqhttp_source"]["accepted_count"] == 1
        assert health["aiocqhttp_source"]["duplicate_count"] == 1


class TestPluginTerminate:
    """Tests for plugin lifecycle management."""

    @pytest.mark.asyncio
    async def test_terminate_closes_connection(self, tmp_path: Path):
        ctx = MagicMock()
        config_mock = MagicMock()
        config_mock.get.side_effect = lambda key, default=None: {"accounts": []}.get(
            key,
            default,
        )
        with patch(
            "astrbot.builtin_stars.qq_clone_observer.main.get_astrbot_plugin_data_path",
            return_value=str(tmp_path),
        ):
            plugin = QQCloneObserver(ctx, config_mock)
            plugin._conn = sqlite3.connect(
                str(tmp_path / "test.db"), check_same_thread=False
            )
            plugin._conn.execute("PRAGMA journal_mode=WAL")
            _build_schema(plugin._conn.cursor())

            assert plugin._conn is not None
            await plugin.terminate()
            assert plugin._conn is None

    @pytest.mark.asyncio
    async def test_terminate_without_conn_does_not_crash(self):
        ctx = MagicMock()
        config_mock = MagicMock()
        plugin = QQCloneObserver(ctx, config_mock)
        plugin._conn = None
        await plugin.terminate()


class TestNormalizedSchema:
    """Tests for normalized_events table schema, columns, and triggers."""

    @pytest.fixture
    def conn(self, tmp_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(str(tmp_path / "test_normalized.db"))
        conn.execute("PRAGMA journal_mode=WAL")
        _build_schema(conn.cursor())
        _build_normalized_schema(conn.cursor())
        from astrbot.builtin_stars.qq_clone_observer.main import _insert_raw_event

        _insert_raw_event(
            conn,
            "p1",
            "bot1",
            "source",
            "message",
            None,
            "msg_001",
            "incoming",
            "qq",
            "{}",
            "sha",
            "fp",
            "key1",
        )
        return conn

    def test_normalized_table_exists(self, conn: sqlite3.Connection):
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='normalized_events'",
        )
        assert cursor.fetchone() is not None

    def test_normalized_columns_exist(self, conn: sqlite3.Connection):
        cursor = conn.execute("PRAGMA table_info(normalized_events)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            "id",
            "raw_event_id",
            "parser_version",
            "platform_id",
            "self_id",
            "account_role",
            "source_event_type",
            "source_event_id",
            "source_message_id",
            "conversation_type",
            "conversation_id",
            "sender_id",
            "sender_display_name",
            "group_id",
            "direction",
            "provenance",
            "actor_provenance",
            "normalized_segments_json",
            "reply_target_message_id",
            "source_occurred_time",
            "normalized_processing_time",
            "normalization_status",
            "normalization_error",
        }
        assert expected.issubset(columns)

    def test_unique_raw_event_id(self, conn: sqlite3.Connection):
        _insert_normalized_event(
            conn,
            {
                "raw_event_id": 1,
                "parser_version": "0.1.0",
                "platform_id": "p1",
                "self_id": "bot1",
                "account_role": "source",
                "source_event_type": None,
                "source_event_id": None,
                "source_message_id": None,
                "conversation_type": "private",
                "conversation_id": "p1:bot1:p:user1",
                "sender_id": None,
                "sender_display_name": None,
                "group_id": None,
                "direction": "incoming",
                "provenance": None,
                "actor_provenance": "unknown",
                "normalized_segments_json": "[]",
                "reply_target_message_id": None,
                "source_occurred_time": None,
                "normalized_processing_time": "1000",
                "normalization_status": "normalized",
                "normalization_error": None,
            },
        )
        inserted = _insert_normalized_event(
            conn,
            {
                "raw_event_id": 1,
                "parser_version": "0.1.0",
                "platform_id": "p1",
                "self_id": "bot1",
                "account_role": "source",
                "source_event_type": None,
                "source_event_id": None,
                "source_message_id": None,
                "conversation_type": "private",
                "conversation_id": "p1:bot1:p:user1",
                "sender_id": None,
                "sender_display_name": None,
                "group_id": None,
                "direction": "incoming",
                "provenance": None,
                "actor_provenance": "unknown",
                "normalized_segments_json": "[]",
                "reply_target_message_id": None,
                "source_occurred_time": None,
                "normalized_processing_time": "1000",
                "normalization_status": "normalized",
                "normalization_error": None,
            },
        )
        assert inserted is False

    def test_normalized_update_trigger_rejects_update(self, conn: sqlite3.Connection):
        _insert_normalized_event(
            conn,
            {
                "raw_event_id": 1,
                "parser_version": "0.1.0",
                "platform_id": "p1",
                "self_id": "bot1",
                "account_role": "source",
                "source_event_type": None,
                "source_event_id": None,
                "source_message_id": None,
                "conversation_type": "private",
                "conversation_id": "p1:bot1:p:user1",
                "sender_id": None,
                "sender_display_name": None,
                "group_id": None,
                "direction": "incoming",
                "provenance": None,
                "actor_provenance": "unknown",
                "normalized_segments_json": "[]",
                "reply_target_message_id": None,
                "source_occurred_time": None,
                "normalized_processing_time": "1000",
                "normalization_status": "normalized",
                "normalization_error": None,
            },
        )
        with pytest.raises(sqlite3.DatabaseError, match="UPDATE"):
            conn.execute("UPDATE normalized_events SET normalization_status = 'error'")

    def test_normalized_delete_trigger_rejects_delete(self, conn: sqlite3.Connection):
        _insert_normalized_event(
            conn,
            {
                "raw_event_id": 1,
                "parser_version": "0.1.0",
                "platform_id": "p1",
                "self_id": "bot1",
                "account_role": "source",
                "source_event_type": None,
                "source_event_id": None,
                "source_message_id": None,
                "conversation_type": "private",
                "conversation_id": "p1:bot1:p:user1",
                "sender_id": None,
                "sender_display_name": None,
                "group_id": None,
                "direction": "incoming",
                "provenance": None,
                "actor_provenance": "unknown",
                "normalized_segments_json": "[]",
                "reply_target_message_id": None,
                "source_occurred_time": None,
                "normalized_processing_time": "1000",
                "normalization_status": "normalized",
                "normalization_error": None,
            },
        )
        with pytest.raises(sqlite3.DatabaseError, match="DELETE"):
            conn.execute("DELETE FROM normalized_events")

    def test_insert_and_select_normalized(self, conn: sqlite3.Connection):
        _insert_normalized_event(
            conn,
            {
                "raw_event_id": 1,
                "parser_version": "0.1.0",
                "platform_id": "p1",
                "self_id": "bot1",
                "account_role": "source",
                "source_event_type": "message",
                "source_event_id": "evt_001",
                "source_message_id": "msg_001",
                "conversation_type": "group",
                "conversation_id": "p1:bot1:g:g123",
                "sender_id": "user_001",
                "sender_display_name": "Alice",
                "group_id": "g123",
                "direction": "incoming",
                "provenance": "qq",
                "actor_provenance": "unknown",
                "normalized_segments_json": '[{"kind": "text", "data": {"text": "hello"}}]',
                "reply_target_message_id": "reply_001",
                "source_occurred_time": 1000,
                "normalized_processing_time": "2000",
                "normalization_status": "normalized",
                "normalization_error": None,
            },
        )
        cursor = conn.execute("SELECT * FROM normalized_events")
        row = cursor.fetchone()
        assert row is not None
        assert row[1] == 1
        assert row[2] == "0.1.0"
        assert row[3] == "p1"
        assert row[4] == "bot1"
        assert row[5] == "source"
        assert row[9] == "group"
        assert row[10] == "p1:bot1:g:g123"
        assert row[11] == "user_001"
        assert row[13] == "g123"
        assert row[14] == "incoming"
        assert row[16] == "unknown"
        assert row[18] == "reply_001"
        assert row[19] == 1000
        assert row[21] == "normalized"


class TestNormalizedSegmentConversion:
    """Tests for _normalize_segment and _normalize_segments."""

    def test_plain_text_segment(self):
        from astrbot.core.message.components import Plain

        result = _normalize_segment(Plain(text="hello world"))
        assert result["kind"] == "text"
        assert result["data"]["text"] == "hello world"

    def test_at_segment(self):
        from astrbot.core.message.components import At

        comp = At(qq="123456", name="User")
        result = _normalize_segment(comp)
        assert result["kind"] == "at"
        assert result["data"]["qq"] == "123456"
        assert result["data"]["name"] == "User"

    def test_reply_segment(self):
        from astrbot.core.message.components import Reply

        comp = Reply(id="msg_001")
        result = _normalize_segment(comp)
        assert result["kind"] == "reply"
        assert result["data"]["id"] == "msg_001"

    def test_image_segment(self):
        from astrbot.core.message.components import Image

        comp = Image(file="test.jpg", url="http://example.com/img.jpg")
        result = _normalize_segment(comp)
        assert result["kind"] == "image"
        assert "test.jpg" in result["data"]["file"]

    def test_record_segment(self):
        from astrbot.core.message.components import Record

        comp = Record(file="audio.amr", url="http://example.com/audio.amr")
        result = _normalize_segment(comp)
        assert result["kind"] == "record"
        assert "audio.amr" in result["data"]["file"]

    def test_unknown_segment_preserved(self):
        from astrbot.core.message.components import Unknown

        comp = Unknown(text="unknown_data")
        result = _normalize_segment(comp)
        assert result["kind"] == "unknown"
        with pytest.raises(KeyError):
            _ = result["data"]["text"]

    def test_ordered_segments(self):
        from astrbot.core.message.components import At, Image, Plain

        msg = MagicMock(spec=AstrBotMessage)
        msg.message = [
            Plain(text="hello"),
            At(qq="123456", name="User"),
            Image(file="pic.jpg"),
        ]
        segments = _normalize_segments(msg)
        assert len(segments) == 3
        assert segments[0]["kind"] == "text"
        assert segments[0]["data"]["text"] == "hello"
        assert segments[1]["kind"] == "at"
        assert segments[1]["data"]["qq"] == "123456"
        assert segments[2]["kind"] == "image"

    def test_empty_segments(self):
        msg = MagicMock(spec=AstrBotMessage)
        msg.message = []
        assert _normalize_segments(msg) == []

    def test_none_segments(self):
        msg = MagicMock(spec=AstrBotMessage)
        msg.message = None
        assert _normalize_segments(msg) == []

    def test_unknown_raw_segment_preserves_original_fields(self):
        raw_segment = {
            "type": "napcat_extension",
            "data": {"value": 42, "nested": {"enabled": True}},
            "extension_field": "kept",
        }

        assert _normalize_raw_segments({"message": [raw_segment]}) == [
            {
                "kind": "unknown",
                "data": {
                    "original_type": "napcat_extension",
                    "original_segment": raw_segment,
                },
            }
        ]

    def test_known_raw_segment_preserves_unknown_data_fields(self):
        assert _normalize_raw_segments(
            {
                "message": [
                    {
                        "type": "image",
                        "data": {"file": "a.jpg", "napcat_extra": "kept"},
                        "extension_field": "also kept",
                    }
                ]
            }
        ) == [
            {
                "kind": "image",
                "data": {
                    "file": "a.jpg",
                    "napcat_extra": "kept",
                    "_extra_fields": {"extension_field": "also kept"},
                },
            }
        ]


class TestConversationTypeDerivation:
    """Tests for _derive_conversation_type."""

    def test_group_from_raw_message(self):
        msg = MagicMock(spec=AstrBotMessage)
        raw = {"message_type": "group"}
        assert _derive_conversation_type(msg, raw) == "group"

    def test_private_from_raw_message(self):
        msg = MagicMock(spec=AstrBotMessage)
        raw = {"message_type": "private"}
        assert _derive_conversation_type(msg, raw) == "private"

    def test_group_from_message_type_enum(self):
        msg = MagicMock(spec=AstrBotMessage)
        msg.type = MessageType.GROUP_MESSAGE
        assert _derive_conversation_type(msg, None) == "group"

    def test_private_from_message_type_enum(self):
        msg = MagicMock(spec=AstrBotMessage)
        msg.type = MessageType.FRIEND_MESSAGE
        assert _derive_conversation_type(msg, None) == "private"

    def test_group_from_group_object(self):
        msg = MagicMock(spec=AstrBotMessage)
        msg.type = MessageType.OTHER_MESSAGE
        msg.group = Group(group_id="g123")
        assert _derive_conversation_type(msg, {"message_type": "other"}) == "group"

    def test_unknown_when_no_evidence(self):
        msg = MagicMock(spec=AstrBotMessage)
        msg.type = MessageType.OTHER_MESSAGE
        msg.group = None
        assert _derive_conversation_type(msg, {}) == "unknown"

    def test_raw_message_get_exception_safe(self):
        class BadGet:
            def get(self, key, default=None):
                raise TypeError("bad")

        msg = MagicMock(spec=AstrBotMessage)
        msg.type = MessageType.OTHER_MESSAGE
        assert _derive_conversation_type(msg, BadGet()) == "unknown"


class TestConversationIdBuilding:
    """Tests for _build_conversation_id."""

    def test_group_conversation_id(self):
        msg = MagicMock(spec=AstrBotMessage)
        msg.group = Group(group_id="g123")
        cid = _build_conversation_id("p1", "bot1", "group", msg, {})
        assert cid == "p1:bot1:g:g123"

    def test_group_conversation_id_from_raw(self):
        msg = MagicMock(spec=AstrBotMessage)
        msg.group = None
        cid = _build_conversation_id("p1", "bot1", "group", msg, {"group_id": "g456"})
        assert cid == "p1:bot1:g:g456"

    def test_group_unknown_when_no_id(self):
        msg = MagicMock(spec=AstrBotMessage)
        msg.group = None
        cid = _build_conversation_id("p1", "bot1", "group", msg, {})
        assert cid == "p1:bot1:g:unknown"

    def test_private_incoming_conversation_id(self):
        msg = MagicMock(spec=AstrBotMessage)
        msg.direction = MessageDirection.INCOMING
        msg.sender = MessageMember(user_id="user_001")
        cid = _build_conversation_id("p1", "bot1", "private", msg, {})
        assert cid == "p1:bot1:p:user_001"

    def test_private_outgoing_conversation_id(self):
        msg = MagicMock(spec=AstrBotMessage)
        msg.direction = MessageDirection.OUTGOING
        msg.sender = MessageMember(user_id="user_001")
        cid = _build_conversation_id(
            "p1", "bot1", "private", msg, {"user_id": "target_001"}
        )
        assert cid == "p1:bot1:p:target_001"

    def test_private_unknown_when_no_peer(self):
        msg = MagicMock(spec=AstrBotMessage)
        msg.direction = MessageDirection.UNKNOWN
        msg.sender = None
        cid = _build_conversation_id("p1", "bot1", "private", msg, {})
        assert cid == "p1:bot1:p:unknown"

    def test_unknown_conversation_id(self):
        msg = MagicMock(spec=AstrBotMessage)
        cid = _build_conversation_id("p1", "bot1", "unknown", msg, {})
        assert cid == "p1:bot1:u:unknown"

    def test_no_collision_between_private_and_group(self):
        msg_private = MagicMock(spec=AstrBotMessage)
        msg_private.direction = MessageDirection.INCOMING
        msg_private.sender = MessageMember(user_id="123")
        private_id = _build_conversation_id("p1", "bot1", "private", msg_private, {})

        msg_group = MagicMock(spec=AstrBotMessage)
        msg_group.group = Group(group_id="123")
        group_id = _build_conversation_id("p1", "bot1", "group", msg_group, {})

        assert private_id != group_id

    def test_no_collision_between_accounts(self):
        msg = MagicMock(spec=AstrBotMessage)
        msg.group = Group(group_id="g123")
        id1 = _build_conversation_id("p1", "bot1", "group", msg, {})
        id2 = _build_conversation_id("p2", "bot2", "group", msg, {})
        assert id1 != id2

    def test_outgoing_user_id_fallback(self):
        msg = MagicMock(spec=AstrBotMessage)
        msg.direction = MessageDirection.OUTGOING
        msg.sender = MessageMember(user_id="sender_001")
        cid = _build_conversation_id("p1", "bot1", "private", msg, {"other": "val"})
        assert cid == "p1:bot1:p:unknown"


class TestActorProvenance:
    """Tests for _derive_actor_provenance."""

    def test_incoming_is_unknown(self):
        assert _derive_actor_provenance("incoming", "qq", "source") == "unknown"

    def test_unknown_direction_is_unknown(self):
        assert _derive_actor_provenance("unknown", "qq", "source") == "unknown"

    def test_source_outgoing_no_provenance_human(self):
        assert _derive_actor_provenance("outgoing", None, "source") == "human"

    def test_source_outgoing_qq_provenance_human(self):
        assert _derive_actor_provenance("outgoing", "qq", "source") == "human"

    def test_source_outgoing_agent_provenance_unknown(self):
        assert _derive_actor_provenance("outgoing", "agent", "source") == "unknown"

    def test_source_outgoing_generated_provenance_unknown(self):
        assert _derive_actor_provenance("outgoing", "generated", "source") == "unknown"

    def test_source_outgoing_imported_provenance_unknown(self):
        assert _derive_actor_provenance("outgoing", "imported", "source") == "unknown"

    def test_source_outgoing_forwarded_provenance_unknown(self):
        assert _derive_actor_provenance("outgoing", "forwarded", "source") == "unknown"

    def test_source_outgoing_forward_provenance_unknown(self):
        assert _derive_actor_provenance("outgoing", "forward", "source") == "unknown"

    def test_clone_outgoing_explicit_agent_is_agent(self):
        assert _derive_actor_provenance("outgoing", "agent", "clone") == "agent"

    def test_clone_outgoing_explicit_generated_is_agent(self):
        assert _derive_actor_provenance("outgoing", "generated", "clone") == "agent"

    def test_clone_outgoing_qq_is_unknown(self):
        assert _derive_actor_provenance("outgoing", "qq", "clone") == "unknown"

    def test_unknown_role_outgoing_is_unknown(self):
        assert _derive_actor_provenance("outgoing", "qq", "unknown") == "unknown"

    def test_provenance_contains_agent_keyword(self):
        assert _derive_actor_provenance("outgoing", "my_agent_v2", "source") == "human"


class TestNormalizeEvent:
    """Integration tests for _normalize_event."""

    def _make_message_obj(
        self,
        direction: MessageDirection = MessageDirection.INCOMING,
        message_type: MessageType = MessageType.FRIEND_MESSAGE,
        user_id: str = "user_001",
        nickname: str | None = "Alice",
        message_id: str = "msg_001",
        timestamp: int = 1000,
        segments: list | None = None,
        group_id: str | None = None,
    ) -> AstrBotMessage:
        msg = AstrBotMessage()
        msg.type = message_type
        msg.direction = direction
        msg.message_id = message_id
        msg.timestamp = timestamp
        msg.sender = MessageMember(user_id=user_id, nickname=nickname)
        msg.message = segments or []
        msg.self_id = "bot1"
        if group_id:
            msg.group = Group(group_id=group_id)
        return msg

    def test_incoming_private_normalization(self):
        msg = self._make_message_obj()
        raw = {"message_type": "private", "user_id": 123, "time": 999}
        result = _normalize_event(
            1,
            "p1",
            "bot1",
            "source",
            "message",
            None,
            "msg_001",
            "incoming",
            "qq",
            msg,
            raw,
        )
        assert result["raw_event_id"] == 1
        assert result["conversation_type"] == "private"
        assert result["conversation_id"] == "p1:bot1:p:user_001"
        assert result["sender_id"] == "user_001"
        assert result["sender_display_name"] == "Alice"
        assert result["direction"] == "incoming"
        assert result["actor_provenance"] == "unknown"
        assert result["source_occurred_time"] == 999
        assert result["normalization_status"] == "normalized"

    def test_outgoing_private_normalization(self):
        msg = self._make_message_obj(direction=MessageDirection.OUTGOING)
        raw = {"message_type": "private", "user_id": "target_001", "time": 999}
        result = _normalize_event(
            1,
            "p1",
            "bot1",
            "source",
            "message_sent",
            None,
            "msg_002",
            "outgoing",
            "qq",
            msg,
            raw,
        )
        assert result["conversation_type"] == "private"
        assert result["conversation_id"] == "p1:bot1:p:target_001"
        assert result["direction"] == "outgoing"
        assert result["actor_provenance"] == "human"
        assert result["source_occurred_time"] == 999

    def test_incoming_group_normalization(self):
        msg = self._make_message_obj(
            message_type=MessageType.GROUP_MESSAGE,
            group_id="g123",
        )
        raw = {"message_type": "group", "group_id": "g123", "time": 888}
        result = _normalize_event(
            1,
            "p1",
            "bot1",
            "source",
            "message",
            None,
            "msg_003",
            "incoming",
            "qq",
            msg,
            raw,
        )
        assert result["conversation_type"] == "group"
        assert result["conversation_id"] == "p1:bot1:g:g123"
        assert result["group_id"] == "g123"
        assert result["sender_id"] == "user_001"
        assert result["actor_provenance"] == "unknown"

    def test_outgoing_group_normalization(self):
        msg = self._make_message_obj(
            direction=MessageDirection.OUTGOING,
            message_type=MessageType.GROUP_MESSAGE,
            group_id="g456",
        )
        raw = {"message_type": "group", "group_id": "g456"}
        result = _normalize_event(
            1,
            "p1",
            "bot1",
            "source",
            "message_sent",
            None,
            "msg_004",
            "outgoing",
            "qq",
            msg,
            raw,
        )
        assert result["conversation_type"] == "group"
        assert result["conversation_id"] == "p1:bot1:g:g456"
        assert result["group_id"] == "g456"
        assert result["actor_provenance"] == "human"

    def test_source_role_preserved_in_normalized(self):
        msg = self._make_message_obj()
        raw = {"message_type": "private"}
        result = _normalize_event(
            1,
            "p1",
            "bot1",
            "source",
            "message",
            None,
            "msg_001",
            "incoming",
            "qq",
            msg,
            raw,
        )
        assert result["account_role"] == "source"

    def test_clone_role_preserved_in_normalized(self):
        msg = self._make_message_obj()
        raw = {"message_type": "private"}
        result = _normalize_event(
            1,
            "p1",
            "bot1",
            "clone",
            "message",
            None,
            "msg_001",
            "incoming",
            "qq",
            msg,
            raw,
        )
        assert result["account_role"] == "clone"

    def test_normalized_segments_included(self):
        from astrbot.core.message.components import At, Plain

        msg = self._make_message_obj(
            segments=[
                Plain(text="hello"),
                At(qq="123", name="User"),
            ]
        )
        raw = {"message_type": "private"}
        result = _normalize_event(
            1,
            "p1",
            "bot1",
            "source",
            "message",
            None,
            "msg_001",
            "incoming",
            "qq",
            msg,
            raw,
        )
        segments = json.loads(result["normalized_segments_json"])
        assert len(segments) == 2
        assert segments[0]["kind"] == "text"
        assert segments[0]["data"]["text"] == "hello"
        assert segments[1]["kind"] == "at"

    def test_reply_target_extracted(self):
        from astrbot.core.message.components import Plain, Reply

        msg = self._make_message_obj(
            segments=[
                Reply(id="reply_001"),
                Plain(text="hello"),
            ]
        )
        raw = {"message_type": "private"}
        result = _normalize_event(
            1,
            "p1",
            "bot1",
            "source",
            "message",
            None,
            "msg_001",
            "incoming",
            "qq",
            msg,
            raw,
        )
        assert result["reply_target_message_id"] == "reply_001"

    def test_reply_target_from_raw_message(self):
        msg = self._make_message_obj()
        raw = {"message_type": "private", "reply": "raw_reply_001"}
        result = _normalize_event(
            1,
            "p1",
            "bot1",
            "source",
            "message",
            None,
            "msg_001",
            "incoming",
            "qq",
            msg,
            raw,
        )
        assert result["reply_target_message_id"] == "raw_reply_001"

    def test_reply_target_from_raw_segment(self):
        msg = self._make_message_obj()
        raw = {
            "message_type": "private",
            "message": [{"type": "reply", "data": {"id": "raw_reply_002"}}],
        }
        result = _normalize_event(
            1,
            "p1",
            "bot1",
            "source",
            "message",
            None,
            "msg_001",
            "incoming",
            "qq",
            msg,
            raw,
        )
        assert result["reply_target_message_id"] == "raw_reply_002"

    def test_timestamp_from_raw_message(self):
        msg = self._make_message_obj(timestamp=2000)
        raw = {"message_type": "private", "time": 1500}
        result = _normalize_event(
            1,
            "p1",
            "bot1",
            "source",
            "message",
            None,
            "msg_001",
            "incoming",
            "qq",
            msg,
            raw,
        )
        assert result["source_occurred_time"] == 1500

    def test_timestamp_fallback_to_message_obj(self):
        msg = self._make_message_obj(timestamp=2000)
        raw = {"message_type": "private"}
        result = _normalize_event(
            1,
            "p1",
            "bot1",
            "source",
            "message",
            None,
            "msg_001",
            "incoming",
            "qq",
            msg,
            raw,
        )
        assert result["source_occurred_time"] == 2000

    def test_invalid_raw_timestamp_falls_back_to_message_obj(self):
        msg = self._make_message_obj(timestamp=2000)
        raw = {"message_type": "private", "time": "invalid"}
        result = _normalize_event(
            1,
            "p1",
            "bot1",
            "source",
            "message",
            None,
            "msg_001",
            "incoming",
            "qq",
            msg,
            raw,
        )
        assert result["source_occurred_time"] == 2000

    def test_normalization_deterministic(self):
        from astrbot.core.message.components import Plain

        msg = self._make_message_obj(segments=[Plain(text="hello")])
        raw = {"message_type": "private", "time": 1000}
        kwargs = dict(
            raw_event_id=1,
            platform_id="p1",
            self_id="bot1",
            account_role="source",
            source_event_type="message",
            source_event_id="evt_001",
            source_message_id="msg_001",
            direction="incoming",
            provenance="qq",
            message_obj=msg,
            raw_message=raw,
        )
        result1 = _normalize_event(**kwargs)
        result2 = _normalize_event(**kwargs)
        for key in ("normalized_processing_time",):
            result1.pop(key, None)
            result2.pop(key, None)
        assert result1 == result2

    def test_no_sender_info(self):
        msg = AstrBotMessage()
        msg.type = MessageType.FRIEND_MESSAGE
        msg.direction = MessageDirection.INCOMING
        msg.sender = None
        msg.message = []
        msg.timestamp = 1000
        raw = {"message_type": "private"}
        result = _normalize_event(
            1,
            "p1",
            "bot1",
            "source",
            "message",
            None,
            None,
            "incoming",
            "qq",
            msg,
            raw,
        )
        assert result["sender_id"] is None
        assert result["sender_display_name"] is None

    def test_no_message_id(self):
        msg = self._make_message_obj(message_id=None)
        raw = {"message_type": "private"}
        result = _normalize_event(
            1,
            "p1",
            "bot1",
            "source",
            "message",
            None,
            None,
            "incoming",
            "qq",
            msg,
            raw,
        )
        assert result["source_message_id"] is None


class TestNormalizedLayerIntegration:
    """Full integration tests: raw insert triggers normalization."""

    @pytest.fixture
    def plugin_with_db(self, tmp_path: Path):
        ctx = MagicMock()
        config = {
            "accounts": [
                {
                    "platform_id": "aiocqhttp_source",
                    "self_id": "123456",
                    "role": "source",
                    "enabled": True,
                },
                {
                    "platform_id": "aiocqhttp_clone",
                    "self_id": "789012",
                    "role": "clone",
                    "enabled": True,
                },
            ],
        }
        config_mock = MagicMock()
        config_mock.get.side_effect = lambda key, default=None: config.get(key, default)
        with patch(
            "astrbot.builtin_stars.qq_clone_observer.main.get_astrbot_plugin_data_path",
            return_value=str(tmp_path),
        ):
            plugin = QQCloneObserver(ctx, config_mock)
            plugin._conn = sqlite3.connect(
                str(tmp_path / "raw_events.db"), check_same_thread=False
            )
            plugin._conn.execute("PRAGMA journal_mode=WAL")
            _build_schema(plugin._conn.cursor())
            plugin._rebuild_health()
            return plugin

    def _make_event(
        self,
        platform_id: str = "aiocqhttp_source",
        self_id: str = "123456",
        message_id: str = "msg_001",
        direction: MessageDirection = MessageDirection.INCOMING,
        provenance: str = "qq",
        sender_id: str = "user_001",
        sender_nickname: str | None = "Alice",
        raw_message: object = None,
        message_type: MessageType = MessageType.FRIEND_MESSAGE,
        group_id: str | None = None,
        segments: list | None = None,
    ) -> MagicMock:
        event = MagicMock()
        event.get_platform_id.return_value = platform_id
        event.get_self_id.return_value = self_id
        event.get_sender_id.return_value = sender_id

        message_obj = AstrBotMessage()
        message_obj.type = message_type
        message_obj.direction = direction
        message_obj.provenance = provenance
        message_obj.message_id = message_id
        message_obj.timestamp = int(time.time())
        message_obj.raw_message = raw_message
        message_obj.self_id = self_id
        message_obj.sender = MessageMember(user_id=sender_id, nickname=sender_nickname)
        message_obj.message = segments or []
        if group_id:
            message_obj.group = Group(group_id=group_id)

        event.message_obj = message_obj
        return event

    @pytest.mark.asyncio
    async def test_raw_insert_creates_normalized_row(self, plugin_with_db):
        raw = {"post_type": "message", "message_type": "private", "self_id": 123456}
        event = self._make_event(raw_message=raw)
        await plugin_with_db.on_message_observed(event)

        cursor = plugin_with_db._conn.execute("SELECT * FROM normalized_events")
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][3] == "aiocqhttp_source"
        assert rows[0][9] == "private"
        assert rows[0][14] == "incoming"
        assert rows[0][16] == "unknown"
        assert rows[0][21] == "normalized"

    @pytest.mark.asyncio
    async def test_normalized_has_raw_event_reference(self, plugin_with_db):
        raw = {"post_type": "message", "message_type": "private"}
        event = self._make_event(raw_message=raw)
        await plugin_with_db.on_message_observed(event)

        cursor = plugin_with_db._conn.execute(
            "SELECT n.raw_event_id, r.id FROM normalized_events n "  # noqa: S608
            "JOIN raw_events r ON n.raw_event_id = r.id"
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == row[1]

    @pytest.mark.asyncio
    async def test_duplicate_delivery_single_normalized(self, plugin_with_db):
        raw = {"post_type": "message", "message_type": "private"}
        event = self._make_event(raw_message=raw)
        await plugin_with_db.on_message_observed(event)
        await plugin_with_db.on_message_observed(event)

        cursor = plugin_with_db._conn.execute("SELECT COUNT(*) FROM raw_events")
        assert cursor.fetchone()[0] == 1
        cursor = plugin_with_db._conn.execute("SELECT COUNT(*) FROM normalized_events")
        assert cursor.fetchone()[0] == 1

    @pytest.mark.asyncio
    async def test_incoming_group_normalized(self, plugin_with_db):
        raw = {"post_type": "message", "message_type": "group", "group_id": "g123"}
        event = self._make_event(
            raw_message=raw,
            message_type=MessageType.GROUP_MESSAGE,
            group_id="g123",
        )
        await plugin_with_db.on_message_observed(event)

        cursor = plugin_with_db._conn.execute("SELECT * FROM normalized_events")
        row = cursor.fetchone()
        assert row[9] == "group"
        assert row[10] == "aiocqhttp_source:123456:g:g123"
        assert row[13] == "g123"

    @pytest.mark.asyncio
    async def test_outgoing_group_normalized(self, plugin_with_db):
        raw = {"post_type": "message_sent", "message_type": "group", "group_id": "g456"}
        event = self._make_event(
            raw_message=raw,
            direction=MessageDirection.OUTGOING,
            message_type=MessageType.GROUP_MESSAGE,
            group_id="g456",
        )
        await plugin_with_db.on_message_observed(event)

        cursor = plugin_with_db._conn.execute("SELECT * FROM normalized_events")
        row = cursor.fetchone()
        assert row[9] == "group"
        assert row[14] == "outgoing"
        assert row[16] == "human"

    @pytest.mark.asyncio
    async def test_source_outgoing_actor_provenance_human(self, plugin_with_db):
        raw = {"post_type": "message_sent", "message_type": "private"}
        event = self._make_event(
            raw_message=raw,
            direction=MessageDirection.OUTGOING,
        )
        await plugin_with_db.on_message_observed(event)

        cursor = plugin_with_db._conn.execute(
            "SELECT actor_provenance FROM normalized_events"
        )
        assert cursor.fetchone()[0] == "human"

    @pytest.mark.asyncio
    async def test_clone_role_normalized(self, plugin_with_db):
        raw = {"post_type": "message", "message_type": "private"}
        event = self._make_event(
            platform_id="aiocqhttp_clone",
            self_id="789012",
            raw_message=raw,
            direction=MessageDirection.OUTGOING,
        )
        await plugin_with_db.on_message_observed(event)

        cursor = plugin_with_db._conn.execute(
            "SELECT account_role, actor_provenance FROM normalized_events"
        )
        row = cursor.fetchone()
        assert row[0] == "clone"
        assert row[1] == "unknown"

    @pytest.mark.asyncio
    async def test_segments_in_normalized(self, plugin_with_db):
        from astrbot.core.message.components import At, Plain

        raw = {
            "post_type": "message",
            "message_type": "private",
            "message": [
                {"type": "text", "data": {"text": "hello"}},
                {"type": "at", "data": {"qq": "123"}},
            ],
        }
        event = self._make_event(
            raw_message=raw,
            segments=[Plain(text="hello"), At(qq="123", name="User")],
        )
        await plugin_with_db.on_message_observed(event)

        cursor = plugin_with_db._conn.execute(
            "SELECT normalized_segments_json FROM normalized_events"
        )
        segments = json.loads(cursor.fetchone()[0])
        assert len(segments) == 2
        assert segments[0]["kind"] == "text"
        assert segments[0]["data"]["text"] == "hello"
        assert segments[1]["kind"] == "at"

    @pytest.mark.asyncio
    async def test_reply_target_in_normalized(self, plugin_with_db):
        from astrbot.core.message.components import Plain, Reply

        raw = {"post_type": "message", "message_type": "private"}
        event = self._make_event(
            raw_message=raw,
            segments=[Reply(id="reply_001"), Plain(text="hello")],
        )
        await plugin_with_db.on_message_observed(event)

        cursor = plugin_with_db._conn.execute(
            "SELECT reply_target_message_id FROM normalized_events"
        )
        assert cursor.fetchone()[0] == "reply_001"

    @pytest.mark.asyncio
    async def test_normalization_failure_preserves_raw(self, plugin_with_db):
        """Normalization failure should preserve raw row and not crash."""
        event = self._make_event(
            raw_message={"post_type": "message", "message_type": "private"},
        )
        with patch(
            "astrbot.builtin_stars.qq_clone_observer.main._normalize_event",
            side_effect=ValueError("invalid normalized event"),
        ):
            await plugin_with_db.on_message_observed(event)

        cursor = plugin_with_db._conn.execute("SELECT COUNT(*) FROM raw_events")
        assert cursor.fetchone()[0] == 1

        cursor = plugin_with_db._conn.execute("SELECT COUNT(*) FROM normalized_events")
        assert cursor.fetchone()[0] == 0

        health = plugin_with_db.get_health()["aiocqhttp_source"]
        assert "normalization_error" in health["last_rejection_or_error"]

    @pytest.mark.asyncio
    async def test_plugin_never_sends_after_normalization(self, plugin_with_db):
        raw = {"post_type": "message", "message_type": "private"}
        event = self._make_event(raw_message=raw)
        event.send = MagicMock()
        event.send_streaming = MagicMock()
        event.plain_result = MagicMock()
        event.make_result = MagicMock()
        await plugin_with_db.on_message_observed(event)

        event.send.assert_not_called()
        event.send_streaming.assert_not_called()
        event.plain_result.assert_not_called()
        event.make_result.assert_not_called()
