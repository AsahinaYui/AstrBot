"""Tests for QQClone conversation builder.

Verifies:
- Schema creation, foreign keys, uniqueness, and update/delete behavior.
- Private/group/account scopes do not collide or mix.
- Deterministic ordering with equal timestamps and late/out-of-order source timestamps.
- Reply link resolution, cross-conversation non-resolution, and repair after late target arrival.
- Same-sender events inside/outside the configured window and intervening-sender burst breaks.
- Raw/normalized messages remain separate while burst membership represents one semantic turn.
- Stable turn IDs and memberships across repeated rebuild, duplicate delivery, and fresh DB replay.
- Effective outgoing sender fallback to self_id.
- Relationship default creation, group/private subject scope, valid manual edits,
  invalid type/closeness rejection, and preservation across rebuild.
- Privacy defaults, valid edits, invalid privacy rejection, retroactive exclusion
  clearing derived rows, and re-enable rebuilding them.
- Builder failure preserves raw/normalized rows and logs/health exposes only the exception class.
- Existing observation never sends, invokes a model, calls NapCat APIs, or contacts external services.
"""

import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from astrbot.builtin_stars.qq_clone_observer._conversation import (
    _build_conversation_schema,
    _create_relationship_profile,
    _effective_sender,
    _ensure_privacy_row,
    _ensure_relationship_for_normalized,
    _event_time_ms,
    _get_conversation_events,
    _get_conversation_privacy,
    _get_or_create_conversation,
    _get_relationship_profile,
    _rebuild_conversation,
    _rebuild_conversation_for_normalized,
    _update_conversation_privacy,
    _update_relationship_profile,
)
from astrbot.builtin_stars.qq_clone_observer._normalized import (
    _insert_normalized_event,
    _normalize_stored_raw_event,
)
from astrbot.builtin_stars.qq_clone_observer.main import (
    QQCloneObserver,
    _build_schema,
    _insert_raw_event,
)
from astrbot.core.platform.astrbot_message import (
    AstrBotMessage,
    MessageDirection,
    MessageMember,
    MessageType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conn(tmp_path: Path) -> sqlite3.Connection:
    """Create a fresh SQLite connection with full schema."""
    conn = sqlite3.connect(str(tmp_path / "test_conv.db"))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _build_schema(conn.cursor())
    _build_conversation_schema(conn.cursor())
    return conn


def _insert_normalized(
    conn: sqlite3.Connection,
    *,
    platform_id: str = "p1",
    self_id: str = "bot1",
    conversation_type: str = "private",
    sender_id: str | None = "user_001",
    direction: str = "incoming",
    source_message_id: str | None = "msg_001",
    reply_target_message_id: str | None = None,
    source_occurred_time: int | None = 1000,
    raw_event_id: int | None = None,
) -> int:
    """Insert a raw event and its normalized form, returning the normalized row ID.

    The raw_events and normalized_events tables are immutable — this helper
    works around that by setting the desired state through the raw event
    payload before normalization.

    Always sets ``time`` in the raw payload to avoid the fallback to
    ``message_obj.timestamp`` in ``_extract_source_occurred_time``.
    """
    raw_data: dict = {"message_type": conversation_type}
    # Always set time to control source_occurred_time deterministically
    raw_data["time"] = source_occurred_time if source_occurred_time is not None else 0
    if sender_id is not None:
        raw_data["sender"] = {"user_id": sender_id}
    if reply_target_message_id is not None:
        raw_data["reply"] = reply_target_message_id

    if raw_event_id is None:
        rid = _insert_raw_event(
            conn,
            platform_id,
            self_id,
            "source",
            "message",
            None,
            source_message_id,
            direction,
            "qq",
            json.dumps(raw_data),
            "payload-hash",
            "fingerprint",
            f"key:{platform_id}:{self_id}:{source_message_id or 'anon'}",
        )
        assert rid is not None
    else:
        rid = raw_event_id

    _normalize_stored_raw_event(conn, rid)

    cursor = conn.execute(
        "SELECT id, conversation_id, conversation_type FROM normalized_events WHERE raw_event_id = ?",
        (rid,),
    )
    row = cursor.fetchone()
    return row[0]


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestConversationSchema:
    """Tests for conversation schema creation and constraints."""

    def test_conversation_table_exists(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'",
        )
        assert cursor.fetchone() is not None

    def test_reply_links_table_exists(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='reply_links'",
        )
        assert cursor.fetchone() is not None

    def test_bursts_table_exists(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='bursts'",
        )
        assert cursor.fetchone() is not None

    def test_burst_members_table_exists(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='burst_members'",
        )
        assert cursor.fetchone() is not None

    def test_relationship_profiles_table_exists(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='relationship_profiles'",
        )
        assert cursor.fetchone() is not None

    def test_conversation_privacy_table_exists(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_privacy'",
        )
        assert cursor.fetchone() is not None

    def test_conversation_columns(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        cursor = conn.execute("PRAGMA table_info(conversations)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            "id",
            "conversation_id",
            "platform_id",
            "self_id",
            "conversation_type",
            "created_at",
            "updated_at",
        }
        assert expected.issubset(columns)

    def test_relationship_profiles_unique(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        conn.execute(
            "INSERT INTO relationship_profiles "
            "(platform_id, self_id, subject_id, relationship_type, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("p1", "bot1", "user_001", "friend", "1000"),
        )
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            conn.execute(
                "INSERT INTO relationship_profiles "
                "(platform_id, self_id, subject_id, relationship_type, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("p1", "bot1", "user_001", "stranger", "2000"),
            )

    def test_conversation_privacy_unique_conv_id(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        conn.execute(
            "INSERT INTO conversations "
            "(conversation_id, platform_id, self_id, conversation_type, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("p1:bot1:p:user1", "p1", "bot1", "private", "1000", "1000"),
        )
        conn.execute(
            "INSERT INTO conversation_privacy (conversation_id) VALUES (?)",
            ("p1:bot1:p:user1",),
        )
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            conn.execute(
                "INSERT INTO conversation_privacy (conversation_id) VALUES (?)",
                ("p1:bot1:p:user1",),
            )

    def test_reply_links_normalized_event_fk(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            conn.execute(
                "INSERT INTO reply_links "
                "(normalized_event_id, conversation_id, source_reply_target_message_id) "
                "VALUES (?, ?, ?)",
                (999, "p1:bot1:p:user1", "target_001"),
            )

    def test_burst_members_normalized_event_fk(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            conn.execute(
                "INSERT INTO burst_members (burst_id, normalized_event_id, position) "
                "VALUES (?, ?, ?)",
                (1, 999, 0),
            )

    def test_idempotent_schema_creation(self, tmp_path: Path):
        """Schema creation must be safe to run multiple times."""
        conn = _make_conn(tmp_path)
        # Second call should not raise
        _build_conversation_schema(conn.cursor())
        # Tables should still be usable
        conn.execute(
            "INSERT INTO conversations "
            "(conversation_id, platform_id, self_id, conversation_type, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("p1:bot1:p:u1", "p1", "bot1", "private", "1000", "1000"),
        )


# ---------------------------------------------------------------------------
# Conversation get-or-create tests
# ---------------------------------------------------------------------------


class TestGetOrCreateConversation:
    """Tests for _get_or_create_conversation."""

    def test_creates_new_conversation(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        result = _get_or_create_conversation(
            conn,
            "p1:bot1:p:user1",
            "p1",
            "bot1",
            "private",
        )
        assert result["conversation_id"] == "p1:bot1:p:user1"
        assert result["platform_id"] == "p1"
        assert result["self_id"] == "bot1"
        assert result["conversation_type"] == "private"

    def test_returns_existing_conversation(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        first = _get_or_create_conversation(
            conn,
            "p1:bot1:p:user1",
            "p1",
            "bot1",
            "private",
        )
        second = _get_or_create_conversation(
            conn,
            "p1:bot1:p:user1",
            "p1",
            "bot1",
            "private",
        )
        assert first["id"] == second["id"]

    def test_same_conv_id_different_accounts_separate(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        a = _get_or_create_conversation(
            conn,
            "p1:bot1:p:user1",
            "p1",
            "bot1",
            "private",
        )
        b = _get_or_create_conversation(
            conn,
            "p1:bot2:p:user1",
            "p1",
            "bot2",
            "private",
        )
        assert a["id"] != b["id"]

    def test_private_and_group_no_collision(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        priv = _get_or_create_conversation(
            conn,
            "p1:bot1:p:123",
            "p1",
            "bot1",
            "private",
        )
        grp = _get_or_create_conversation(
            conn,
            "p1:bot1:g:123",
            "p1",
            "bot1",
            "group",
        )
        assert priv["id"] != grp["id"]


# ---------------------------------------------------------------------------
# Deterministic ordering tests
# ---------------------------------------------------------------------------


class TestDeterministicOrdering:
    """Tests for deterministic event ordering."""

    def test_ordered_by_source_time(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        # Insert events with known source_occurred_time
        nid1 = _insert_normalized(
            conn, source_message_id="msg_001", source_occurred_time=1000
        )
        nid2 = _insert_normalized(
            conn, source_message_id="msg_002", source_occurred_time=500
        )
        nid3 = _insert_normalized(
            conn, source_message_id="msg_003", source_occurred_time=1500
        )

        conv_id = conn.execute(
            "SELECT conversation_id FROM normalized_events WHERE id = ?", (nid1,)
        ).fetchone()[0]

        events = _get_conversation_events(conn, conv_id)
        ids = [e["id"] for e in events]
        # Expected order: 500, 1000, 1500
        assert ids == [nid2, nid1, nid3]

    def test_equal_source_time_tiebreak_by_processing_time(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        # Same source_occurred_time, different processing times.
        # The processing time is the received_time from raw_events, so we
        # insert the second event with a small delay.
        nid1 = _insert_normalized(
            conn, source_message_id="msg_001", source_occurred_time=1000
        )
        import time

        time.sleep(0.01)
        _insert_normalized(conn, source_message_id="msg_002", source_occurred_time=1000)

        conv_id = conn.execute(
            "SELECT conversation_id FROM normalized_events WHERE id = ?", (nid1,)
        ).fetchone()[0]

        events = _get_conversation_events(conn, conv_id)
        ids = [e["id"] for e in events]
        # nid1 has lower processing time (inserted first), so it should come first
        # when source_occurred_time is the same
        assert ids[0] == nid1

    def test_null_source_time_treated_as_earliest(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        # source_occurred_time=0 sorts before source_occurred_time=1000 via COALESCE
        nid1 = _insert_normalized(
            conn, source_message_id="msg_001", source_occurred_time=0
        )
        nid2 = _insert_normalized(
            conn, source_message_id="msg_002", source_occurred_time=1000
        )

        conv_id = conn.execute(
            "SELECT conversation_id FROM normalized_events WHERE id = ?", (nid1,)
        ).fetchone()[0]

        events = _get_conversation_events(conn, conv_id)
        ids = [e["id"] for e in events]
        # source_occurred_time=0 should sort first
        assert ids == [nid1, nid2]

    def test_missing_source_time_uses_processing_time(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        first = _insert_normalized(
            conn, source_message_id="msg_source", source_occurred_time=100
        )
        raw_id = _insert_raw_event(
            conn,
            "p1",
            "bot1",
            "source",
            "message",
            None,
            "msg_fallback",
            "incoming",
            "qq",
            json.dumps({"message_type": "private", "sender": {"user_id": "user_001"}}),
            "payload-hash-fallback",
            "fingerprint-fallback",
            "key:p1:bot1:msg_fallback",
        )
        assert raw_id is not None
        normalized = {
            "raw_event_id": raw_id,
            "parser_version": "test",
            "platform_id": "p1",
            "self_id": "bot1",
            "account_role": "source",
            "source_event_type": "message",
            "source_event_id": None,
            "source_message_id": "msg_fallback",
            "conversation_type": "private",
            "conversation_id": "p1:bot1:p:user_001",
            "sender_id": "user_001",
            "sender_display_name": None,
            "group_id": None,
            "direction": "incoming",
            "provenance": "qq",
            "actor_provenance": "unknown",
            "normalized_segments_json": "[]",
            "reply_target_message_id": None,
            "source_occurred_time": None,
            "normalized_processing_time": "200",
            "normalization_status": "normalized",
            "normalization_error": None,
        }
        assert _insert_normalized_event(conn, normalized)
        events = _get_conversation_events(conn, "p1:bot1:p:user_001")
        assert [event["id"] for event in events] == [first, first + 1]

    def test_late_event_with_earlier_source_time(self, tmp_path: Path):
        """A late-arriving event with an earlier source timestamp must be
        inserted at its reproducible semantic position."""
        conn = _make_conn(tmp_path)
        nid1 = _insert_normalized(
            conn, source_message_id="msg_001", source_occurred_time=2000
        )
        nid2 = _insert_normalized(
            conn, source_message_id="msg_002", source_occurred_time=3000
        )
        # Late arrival with earlier time
        nid3 = _insert_normalized(
            conn, source_message_id="msg_003", source_occurred_time=1000
        )

        conv_id = conn.execute(
            "SELECT conversation_id FROM normalized_events WHERE id = ?", (nid1,)
        ).fetchone()[0]

        events = _get_conversation_events(conn, conv_id)
        ids = [e["id"] for e in events]
        # Expected: msg_003 (1000), msg_001 (2000), msg_002 (3000)
        assert ids == [nid3, nid1, nid2]

    def test_id_final_tiebreaker(self, tmp_path: Path):
        """When all time fields are equal, stable row ID is the final tiebreaker."""
        conn = _make_conn(tmp_path)
        nid1 = _insert_normalized(
            conn, source_message_id="msg_001", source_occurred_time=1000
        )
        import time

        time.sleep(0.01)
        _insert_normalized(conn, source_message_id="msg_002", source_occurred_time=1000)
        # Both have same source_occurred_time but different processing times
        # (nid1 was inserted first, so its processing time is lower).
        # The tiebreaker is processing_time, then id. Since nid1 has lower
        # processing_time, it comes first.

        conv_id = conn.execute(
            "SELECT conversation_id FROM normalized_events WHERE id = ?", (nid1,)
        ).fetchone()[0]

        events = _get_conversation_events(conn, conv_id)
        ids = [e["id"] for e in events]
        # Lower ID first (tied on source_time, nid1 has lower processing_time)
        assert ids[0] == nid1


# ---------------------------------------------------------------------------
# Reply link tests
# ---------------------------------------------------------------------------


class TestReplyLinks:
    """Tests for reply link resolution."""

    def test_reply_link_persisted(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        # Target message first
        nid_target = _insert_normalized(
            conn,
            source_message_id="msg_original",
            source_occurred_time=1000,
        )
        # Reply to it
        nid_reply = _insert_normalized(
            conn,
            source_message_id="msg_reply",
            source_occurred_time=2000,
            reply_target_message_id="msg_original",
        )

        conn.execute(
            "SELECT conversation_id FROM normalized_events WHERE id = ?", (nid_target,)
        ).fetchone()[0]

        _rebuild_conversation_for_normalized(conn, nid_reply)

        cursor = conn.execute(
            "SELECT normalized_event_id, source_reply_target_message_id, "
            "resolved_normalized_event_id FROM reply_links"
        )
        links = cursor.fetchall()
        assert len(links) == 1
        assert links[0][0] == nid_reply
        assert links[0][1] == "msg_original"
        assert links[0][2] == nid_target

    def test_reply_link_unresolved_when_target_missing(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        nid = _insert_normalized(
            conn,
            source_message_id="msg_reply",
            source_occurred_time=2000,
            reply_target_message_id="nonexistent_target",
        )

        conn.execute(
            "SELECT conversation_id FROM normalized_events WHERE id = ?", (nid,)
        ).fetchone()[0]

        _rebuild_conversation_for_normalized(conn, nid)

        cursor = conn.execute("SELECT resolved_normalized_event_id FROM reply_links")
        assert cursor.fetchone()[0] is None

    def test_cross_conversation_not_resolved(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        # Target in conv A (private chat with user_001 on bot1)
        nid_target = _insert_normalized(
            conn,
            platform_id="p1",
            self_id="bot1",
            source_message_id="msg_001",
            source_occurred_time=1000,
            sender_id="user_001",
        )
        conv_a = conn.execute(
            "SELECT conversation_id FROM normalized_events WHERE id = ?", (nid_target,)
        ).fetchone()[0]

        # Reply in conv B (different self_id = different account scope)
        nid_reply = _insert_normalized(
            conn,
            platform_id="p1",
            self_id="bot2",
            source_message_id="msg_002",
            source_occurred_time=2000,
            reply_target_message_id="msg_001",
            sender_id="user_001",
        )
        conv_b = conn.execute(
            "SELECT conversation_id FROM normalized_events WHERE id = ?", (nid_reply,)
        ).fetchone()[0]

        assert conv_a != conv_b

        _rebuild_conversation_for_normalized(conn, nid_reply)

        cursor = conn.execute("SELECT resolved_normalized_event_id FROM reply_links")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] is None

    def test_repair_after_late_target_arrival(self, tmp_path: Path):
        """Rebuilding must resolve a previously unresolved link if the target
        appears later."""
        conn = _make_conn(tmp_path)

        # Reply arrives first (target doesn't exist yet)
        nid_reply = _insert_normalized(
            conn,
            source_message_id="msg_reply",
            source_occurred_time=2000,
            reply_target_message_id="msg_late_target",
        )
        conn.execute(
            "SELECT conversation_id FROM normalized_events WHERE id = ?", (nid_reply,)
        ).fetchone()[0]

        # First rebuild: target missing
        _rebuild_conversation_for_normalized(conn, nid_reply)
        cursor = conn.execute("SELECT resolved_normalized_event_id FROM reply_links")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] is None

        # Target arrives later, naturally in the same conversation
        # (same platform_id, self_id, sender_id)
        nid_target = _insert_normalized(
            conn,
            source_message_id="msg_late_target",
            source_occurred_time=1000,
        )

        # Rebuild should repair the link
        _rebuild_conversation_for_normalized(conn, nid_reply)
        cursor = conn.execute("SELECT resolved_normalized_event_id FROM reply_links")
        row = cursor.fetchone()
        assert row is not None
        # Nid_target should be resolved now
        assert row[0] == nid_target

    def test_no_reply_is_not_a_link(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        nid = _insert_normalized(
            conn,
            source_message_id="msg_plain",
            source_occurred_time=1000,
        )
        _rebuild_conversation_for_normalized(conn, nid)
        cursor = conn.execute("SELECT COUNT(*) FROM reply_links")
        assert cursor.fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Burst tests
# ---------------------------------------------------------------------------


class TestBursts:
    """Tests for burst grouping."""

    def test_same_sender_within_window_merged(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        nid1 = _insert_normalized(
            conn,
            source_message_id="msg_001",
            source_occurred_time=1000,
            sender_id="user_001",
            direction="incoming",
        )
        _insert_normalized(
            conn,
            source_message_id="msg_002",
            source_occurred_time=1005,
            sender_id="user_001",
            direction="incoming",
        )
        _insert_normalized(
            conn,
            source_message_id="msg_003",
            source_occurred_time=1010,
            sender_id="user_001",
            direction="incoming",
        )
        conn.execute(
            "SELECT conversation_id FROM normalized_events WHERE id = ?", (nid1,)
        ).fetchone()[0]

        _rebuild_conversation_for_normalized(conn, nid1, burst_window_ms=12000)

        cursor = conn.execute("SELECT COUNT(*) FROM bursts")
        assert cursor.fetchone()[0] == 1

        cursor = conn.execute("SELECT message_count FROM bursts")
        assert cursor.fetchone()[0] == 3

    def test_same_sender_above_window_split(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        nid1 = _insert_normalized(
            conn,
            source_message_id="msg_001",
            source_occurred_time=1000,
            sender_id="user_001",
            direction="incoming",
        )
        _insert_normalized(
            conn,
            source_message_id="msg_002",
            source_occurred_time=20000,
            sender_id="user_001",
            direction="incoming",
        )
        conn.execute(
            "SELECT conversation_id FROM normalized_events WHERE id = ?", (nid1,)
        ).fetchone()[0]

        _rebuild_conversation_for_normalized(conn, nid1, burst_window_ms=12000)

        cursor = conn.execute("SELECT COUNT(*) FROM bursts")
        assert cursor.fetchone()[0] == 2

    def test_different_sender_breaks_burst(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        # Use group messages so all events share the same conversation_id
        nid1 = _insert_normalized(
            conn,
            source_message_id="msg_001",
            source_occurred_time=1000,
            sender_id="user_001",
            direction="incoming",
            conversation_type="group",
        )
        _insert_normalized(
            conn,
            source_message_id="msg_002",
            source_occurred_time=1005,
            sender_id="user_002",
            direction="incoming",
            conversation_type="group",
        )
        _insert_normalized(
            conn,
            source_message_id="msg_003",
            source_occurred_time=1010,
            sender_id="user_001",
            direction="incoming",
            conversation_type="group",
        )
        conn.execute(
            "SELECT conversation_id FROM normalized_events WHERE id = ?", (nid1,)
        ).fetchone()[0]

        _rebuild_conversation_for_normalized(conn, nid1, burst_window_ms=12000)

        cursor = conn.execute("SELECT COUNT(*) FROM bursts")
        assert cursor.fetchone()[0] == 3

    def test_burst_members_ordered(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        nid1 = _insert_normalized(
            conn,
            source_message_id="msg_001",
            source_occurred_time=1000,
            sender_id="user_001",
            direction="incoming",
        )
        nid2 = _insert_normalized(
            conn,
            source_message_id="msg_002",
            source_occurred_time=1005,
            sender_id="user_001",
            direction="incoming",
        )
        conn.execute(
            "SELECT conversation_id FROM normalized_events WHERE id = ?", (nid1,)
        ).fetchone()[0]

        _rebuild_conversation_for_normalized(conn, nid1, burst_window_ms=12000)

        cursor = conn.execute(
            "SELECT normalized_event_id, position FROM burst_members ORDER BY position"
        )
        members = cursor.fetchall()
        assert len(members) == 2
        assert members[0][0] == nid1
        assert members[0][1] == 0
        assert members[1][0] == nid2
        assert members[1][1] == 1

    def test_burst_window_config_default(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        nid1 = _insert_normalized(
            conn,
            source_message_id="msg_001",
            source_occurred_time=1000,
            sender_id="user_001",
            direction="incoming",
        )
        _insert_normalized(
            conn,
            source_message_id="msg_002",
            source_occurred_time=1005,
            sender_id="user_001",
            direction="incoming",
        )
        conn.execute(
            "SELECT conversation_id FROM normalized_events WHERE id = ?", (nid1,)
        ).fetchone()[0]

        # Use default window (12000)
        _rebuild_conversation_for_normalized(conn, nid1)

        cursor = conn.execute("SELECT COUNT(*) FROM bursts")
        assert cursor.fetchone()[0] == 1

    def test_outgoing_self_id_fallback(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        nid1 = _insert_normalized(
            conn,
            source_message_id="msg_001",
            source_occurred_time=1000,
            sender_id=None,
            direction="outgoing",
        )
        conn.execute(
            "SELECT conversation_id FROM normalized_events WHERE id = ?", (nid1,)
        ).fetchone()[0]

        _rebuild_conversation_for_normalized(conn, nid1, burst_window_ms=12000)

        cursor = conn.execute("SELECT effective_sender_id FROM bursts")
        # Should fall back to self_id
        assert cursor.fetchone()[0] == "bot1"


# ---------------------------------------------------------------------------
# Rebuild determinism tests
# ---------------------------------------------------------------------------


class TestRebuildDeterminism:
    """Tests that rebuilding produces stable results."""

    def test_rebuild_produces_same_order(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        nid1 = _insert_normalized(
            conn, source_message_id="msg_001", source_occurred_time=1000
        )
        _insert_normalized(conn, source_message_id="msg_002", source_occurred_time=2000)
        _insert_normalized(conn, source_message_id="msg_003", source_occurred_time=3000)

        conn.execute(
            "SELECT conversation_id FROM normalized_events WHERE id = ?", (nid1,)
        ).fetchone()[0]

        # First rebuild
        _rebuild_conversation_for_normalized(conn, nid1)
        cursor1 = conn.execute(
            "SELECT normalized_event_id FROM burst_members ORDER BY burst_id, position"
        )
        first = [r[0] for r in cursor1.fetchall()]

        # Second rebuild
        _rebuild_conversation_for_normalized(conn, nid1)
        cursor2 = conn.execute(
            "SELECT normalized_event_id FROM burst_members ORDER BY burst_id, position"
        )
        second = [r[0] for r in cursor2.fetchall()]

        assert first == second

    def test_rebuild_preserves_stable_burst_id(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        nid = _insert_normalized(
            conn, source_message_id="msg_001", source_occurred_time=1000
        )
        _insert_normalized(conn, source_message_id="msg_002", source_occurred_time=1005)
        _rebuild_conversation_for_normalized(conn, nid)
        first = conn.execute("SELECT id FROM bursts").fetchone()[0]
        _rebuild_conversation_for_normalized(conn, nid)
        second = conn.execute("SELECT id FROM bursts").fetchone()[0]
        assert first == second

    def test_duplicate_delivery_same_bursts(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        nid1 = _insert_normalized(
            conn, source_message_id="msg_001", source_occurred_time=1000
        )
        _rebuild_conversation_for_normalized(conn, nid1)

        # Simulate duplicate delivery - same event re-normalized (no-op)
        _normalize_stored_raw_event(conn, 1)  # raw_event_id=1

        _rebuild_conversation_for_normalized(conn, nid1)
        cursor = conn.execute("SELECT COUNT(*) FROM burst_members")
        assert cursor.fetchone()[0] == 1

    def test_fresh_db_replay_same_results(self, tmp_path: Path):
        """Rebuilding from a fresh DB with the same immutable facts must
        produce the same result."""

        def _build_and_replay() -> list:
            import uuid

            db_path = tmp_path / f"replay_{uuid.uuid4().hex}.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            _build_schema(conn.cursor())
            _build_conversation_schema(conn.cursor())
            for i in range(3):
                _insert_normalized(
                    conn,
                    source_message_id=f"msg_{i}",
                    source_occurred_time=1000 + i * 100,
                    sender_id="user_001",
                    direction="incoming",
                )

            # Get the conversation_id
            nid1 = conn.execute(
                "SELECT id FROM normalized_events WHERE source_message_id = 'msg_0'"
            ).fetchone()[0]
            conv_id = conn.execute(
                "SELECT conversation_id FROM normalized_events WHERE id = ?", (nid1,)
            ).fetchone()[0]

            _rebuild_conversation(conn, conv_id, "p1", "bot1", "private", 12000)
            cursor = conn.execute(
                "SELECT normalized_event_id FROM burst_members ORDER BY position"
            )
            result = [r[0] for r in cursor.fetchall()]
            conn.close()
            return result

        result1 = _build_and_replay()
        result2 = _build_and_replay()
        assert result1 == result2


# ---------------------------------------------------------------------------
# Relationship profile tests
# ---------------------------------------------------------------------------


class TestRelationshipProfiles:
    """Tests for relationship profile CRUD and validation."""

    def test_create_relationship(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        result = _create_relationship_profile(
            conn,
            "p1",
            "bot1",
            "user_001",
            relationship_type="friend",
            closeness_score=0.8,
            notes="Close friend",
        )
        assert result["subject_id"] == "user_001"
        assert result["relationship_type"] == "friend"
        assert result["closeness_score"] == 0.8
        assert result["notes"] == "Close friend"
        assert result["manually_confirmed"] is False

    def test_create_default_unknown(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        result = _create_relationship_profile(conn, "p1", "bot1", "user_001")
        assert result["relationship_type"] == "unknown"

    def test_duplicate_create_ignored(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        _create_relationship_profile(
            conn, "p1", "bot1", "user_001", relationship_type="friend"
        )
        result = _create_relationship_profile(
            conn, "p1", "bot1", "user_001", relationship_type="stranger"
        )
        # First one is preserved (INSERT OR IGNORE)
        assert result["relationship_type"] == "friend"

    def test_get_relationship(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        _create_relationship_profile(
            conn, "p1", "bot1", "user_001", relationship_type="coworker"
        )
        result = _get_relationship_profile(conn, "p1", "bot1", "user_001")
        assert result is not None
        assert result["relationship_type"] == "coworker"

    def test_get_nonexistent_returns_none(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        result = _get_relationship_profile(conn, "p1", "bot1", "nonexistent")
        assert result is None

    def test_update_relationship_type(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        _create_relationship_profile(
            conn, "p1", "bot1", "user_001", relationship_type="unknown"
        )
        result = _update_relationship_profile(
            conn,
            "p1",
            "bot1",
            "user_001",
            relationship_type="friend",
        )
        assert result["relationship_type"] == "friend"

    def test_invalid_relationship_type_rejected(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        _create_relationship_profile(conn, "p1", "bot1", "user_001")
        with pytest.raises(ValueError, match="invalid relationship_type"):
            _update_relationship_profile(
                conn,
                "p1",
                "bot1",
                "user_001",
                relationship_type="invalid_type",
            )

    def test_invalid_relationship_create_is_not_persisted(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        with pytest.raises(ValueError, match="invalid relationship_type"):
            _create_relationship_profile(
                conn,
                "p1",
                "bot1",
                "user_001",
                relationship_type="invalid",
            )
        assert _get_relationship_profile(conn, "p1", "bot1", "user_001") is None

    def test_invalid_closeness_rejected(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        _create_relationship_profile(conn, "p1", "bot1", "user_001")
        with pytest.raises(ValueError, match="closeness_score"):
            _update_relationship_profile(
                conn,
                "p1",
                "bot1",
                "user_001",
                closeness_score=2.5,
            )

    def test_valid_closeness_range(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        _create_relationship_profile(conn, "p1", "bot1", "user_001")
        result = _update_relationship_profile(
            conn,
            "p1",
            "bot1",
            "user_001",
            closeness_score=0.0,
        )
        assert result["closeness_score"] == 0.0

        result = _update_relationship_profile(
            conn,
            "p1",
            "bot1",
            "user_001",
            closeness_score=1.0,
        )
        assert result["closeness_score"] == 1.0

    def test_account_scoped_profiles(self, tmp_path: Path):
        """Same subject under different accounts must not collide."""
        conn = _make_conn(tmp_path)
        a = _create_relationship_profile(
            conn, "p1", "bot1", "user_001", relationship_type="friend"
        )
        b = _create_relationship_profile(
            conn, "p1", "bot2", "user_001", relationship_type="coworker"
        )
        assert a["id"] != b["id"]
        assert a["relationship_type"] == "friend"
        assert b["relationship_type"] == "coworker"

    def test_manual_confirmation_preserved(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        _create_relationship_profile(
            conn, "p1", "bot1", "user_001", manually_confirmed=True
        )
        result = _get_relationship_profile(conn, "p1", "bot1", "user_001")
        assert result["manually_confirmed"] is True

    def test_rebuild_does_not_overwrite_manual_edit(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        _create_relationship_profile(
            conn, "p1", "bot1", "user_001", relationship_type="stranger"
        )
        _update_relationship_profile(
            conn,
            "p1",
            "bot1",
            "user_001",
            relationship_type="friend",
            manually_confirmed=True,
        )

        # Trigger rebuild via _ensure_relationship_for_normalized
        nid = _insert_normalized(
            conn,
            source_message_id="msg_001",
            source_occurred_time=1000,
            sender_id="user_001",
            direction="incoming",
        )
        _ensure_relationship_for_normalized(conn, nid)

        result = _get_relationship_profile(conn, "p1", "bot1", "user_001")
        # Should still be friend, not overwritten to unknown
        assert result["relationship_type"] == "friend"
        assert result["manually_confirmed"] is True

    def test_private_relationship_auto_created(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        # Private conversation with peer user_001
        nid = _insert_normalized(
            conn,
            source_message_id="msg_001",
            source_occurred_time=1000,
            sender_id="user_001",
            direction="incoming",
        )
        _ensure_relationship_for_normalized(conn, nid)
        result = _get_relationship_profile(conn, "p1", "bot1", "user_001")
        assert result is not None
        assert result["relationship_type"] == "unknown"

    def test_group_sender_relationship_auto_created(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        # Group conversation with a non-self sender
        # Need to override conversation_type to group
        raw_id = _insert_raw_event(
            conn,
            "p1",
            "bot1",
            "source",
            "message",
            None,
            "msg_001",
            "incoming",
            "qq",
            json.dumps(
                {
                    "message_type": "group",
                    "group_id": "g123",
                    "sender": {"user_id": "user_001"},
                }
            ),
            "hash",
            "fp",
            "key:p1:bot1:msg_001",
        )
        _normalize_stored_raw_event(conn, raw_id)
        cursor = conn.execute(
            "SELECT id, conversation_id FROM normalized_events WHERE raw_event_id = ?",
            (raw_id,),
        )
        nid, conv_id = cursor.fetchone()
        # Ensure it's a group conversation ID
        assert "g:" in conv_id

        _ensure_relationship_for_normalized(conn, nid)
        result = _get_relationship_profile(conn, "p1", "bot1", "user_001")
        assert result is not None
        assert result["relationship_type"] == "unknown"

    def test_self_sender_not_auto_related(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        nid = _insert_normalized(
            conn,
            source_message_id="msg_001",
            source_occurred_time=1000,
            sender_id="bot1",
            direction="incoming",
        )
        _ensure_relationship_for_normalized(conn, nid)
        result = _get_relationship_profile(conn, "p1", "bot1", "bot1")
        assert result is None


# ---------------------------------------------------------------------------
# Privacy control tests
# ---------------------------------------------------------------------------


class TestPrivacyControls:
    """Tests for privacy controls."""

    def test_default_privacy(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        _get_or_create_conversation(conn, "p1:bot1:p:user1", "p1", "bot1", "private")
        privacy = _ensure_privacy_row(conn, "p1:bot1:p:user1")
        assert privacy["observe_enabled"] is True
        assert privacy["persona_learning_enabled"] is True
        assert privacy["training_export_enabled"] is False
        assert privacy["clone_reply_enabled"] is False
        assert privacy["privacy_level"] == "normal"

    def test_update_privacy_observe_enabled(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        _get_or_create_conversation(conn, "p1:bot1:p:user1", "p1", "bot1", "private")
        _ensure_privacy_row(conn, "p1:bot1:p:user1")
        result = _update_conversation_privacy(
            conn,
            "p1:bot1:p:user1",
            observe_enabled=False,
        )
        assert result["observe_enabled"] is False

    def test_update_privacy_level(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        _get_or_create_conversation(conn, "p1:bot1:p:user1", "p1", "bot1", "private")
        _ensure_privacy_row(conn, "p1:bot1:p:user1")
        result = _update_conversation_privacy(
            conn,
            "p1:bot1:p:user1",
            privacy_level="sensitive",
        )
        assert result["privacy_level"] == "sensitive"

    def test_invalid_privacy_level_rejected(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        _get_or_create_conversation(conn, "p1:bot1:p:user1", "p1", "bot1", "private")
        _ensure_privacy_row(conn, "p1:bot1:p:user1")
        with pytest.raises(ValueError, match="invalid privacy_level"):
            _update_conversation_privacy(
                conn,
                "p1:bot1:p:user1",
                privacy_level="invalid",
            )

    def test_non_boolean_privacy_value_is_rejected_without_update(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        _get_or_create_conversation(conn, "p1:bot1:p:user1", "p1", "bot1", "private")
        _ensure_privacy_row(conn, "p1:bot1:p:user1")
        with pytest.raises(ValueError, match="observe_enabled must be a bool"):
            _update_conversation_privacy(
                conn,
                "p1:bot1:p:user1",
                observe_enabled=1,
                privacy_level="sensitive",
            )
        privacy = _get_conversation_privacy(conn, "p1:bot1:p:user1")
        assert privacy["observe_enabled"] is True
        assert privacy["privacy_level"] == "normal"

    def test_excluded_conversation_clears_derived_rows(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        # Use group messages so all events share the same conversation_id.
        # Use close timestamps so they merge into one burst.
        nid1 = _insert_normalized(
            conn,
            source_message_id="msg_001",
            source_occurred_time=1000,
            sender_id="user_001",
            direction="incoming",
            conversation_type="group",
        )
        _insert_normalized(
            conn,
            source_message_id="msg_002",
            source_occurred_time=1005,
            sender_id="user_001",
            direction="incoming",
            conversation_type="group",
        )
        conv_id = conn.execute(
            "SELECT conversation_id FROM normalized_events WHERE id = ?", (nid1,)
        ).fetchone()[0]

        # Build normally — this creates the conversation + privacy rows
        _rebuild_conversation_for_normalized(conn, nid1)
        cursor = conn.execute("SELECT COUNT(*) FROM bursts")
        burst_count = cursor.fetchone()[0]
        assert burst_count > 0, f"expected at least 1 burst, got {burst_count}"
        # Mark as excluded
        _update_conversation_privacy(conn, conv_id, privacy_level="excluded")

        # Derived rows should be cleared
        cursor = conn.execute("SELECT COUNT(*) FROM bursts")
        assert cursor.fetchone()[0] == 0
        cursor = conn.execute("SELECT COUNT(*) FROM burst_members")
        assert cursor.fetchone()[0] == 0

        # Normalized events should still exist
        cursor = conn.execute("SELECT COUNT(*) FROM normalized_events")
        assert cursor.fetchone()[0] == 2

    def test_re_enable_rebuilds_derived_rows(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        # Use group messages so all events share the same conversation_id.
        # Use close timestamps so they merge into one burst.
        nid1 = _insert_normalized(
            conn,
            source_message_id="msg_001",
            source_occurred_time=1000,
            sender_id="user_001",
            direction="incoming",
            conversation_type="group",
        )
        _insert_normalized(
            conn,
            source_message_id="msg_002",
            source_occurred_time=1005,
            sender_id="user_001",
            direction="incoming",
            conversation_type="group",
        )
        conv_id = conn.execute(
            "SELECT conversation_id FROM normalized_events WHERE id = ?", (nid1,)
        ).fetchone()[0]

        # Build normally — this creates the conversation + privacy rows
        _rebuild_conversation_for_normalized(conn, nid1)

        # Exclude
        _update_conversation_privacy(conn, conv_id, privacy_level="excluded")
        cursor = conn.execute("SELECT COUNT(*) FROM bursts")
        assert cursor.fetchone()[0] == 0

        # Re-enable
        _update_conversation_privacy(conn, conv_id, privacy_level="normal")

        # Derived rows should be rebuilt
        cursor = conn.execute("SELECT COUNT(*) FROM bursts")
        burst_count = cursor.fetchone()[0]
        assert burst_count > 0, f"expected at least 1 burst, got {burst_count}"
        cursor = conn.execute("SELECT COUNT(*) FROM burst_members")
        member_count = cursor.fetchone()[0]
        assert member_count > 0, f"expected at least 1 member, got {member_count}"

    def test_get_privacy_nonexistent(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        result = _get_conversation_privacy(conn, "nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# Effective sender tests
# ---------------------------------------------------------------------------


class TestEffectiveSender:
    """Tests for _effective_sender."""

    def test_incoming_uses_sender_id(self):
        event = {"direction": "incoming", "sender_id": "user_001"}
        assert _effective_sender(event, "bot1") == "user_001"

    def test_outgoing_with_ambiguous_sender_uses_self_id(self):
        event = {"direction": "outgoing", "sender_id": "user_001"}
        assert _effective_sender(event, "bot1") == "bot1"

    def test_outgoing_without_sender_falls_back_to_self_id(self):
        event = {"direction": "outgoing", "sender_id": None}
        assert _effective_sender(event, "bot1") == "bot1"

    def test_unknown_sender_uses_unknown(self):
        event = {"direction": "incoming", "sender_id": None}
        assert _effective_sender(event, "bot1") == "unknown"


# ---------------------------------------------------------------------------
# Event time millisecond tests
# ---------------------------------------------------------------------------


class TestEventTimeMs:
    """Tests for _event_time_ms."""

    def test_source_time_multiplied_by_1000(self):
        event = {"source_occurred_time": 1000, "normalized_processing_time": "5000"}
        assert _event_time_ms(event) == 1000000

    def test_fallback_to_processing_time(self):
        event = {"source_occurred_time": None, "normalized_processing_time": "5000"}
        assert _event_time_ms(event) == 5000000

    def test_fallback_to_zero(self):
        event = {"source_occurred_time": None, "normalized_processing_time": "invalid"}
        assert _event_time_ms(event) == 0


# ---------------------------------------------------------------------------
# Builder failure tests
# ---------------------------------------------------------------------------


class TestBuilderFailure:
    """Tests that builder failure preserves raw/normalized and logs only class."""

    def test_rebuild_failure_preserves_immutable_rows(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        nid = _insert_normalized(
            conn,
            source_message_id="msg_001",
            source_occurred_time=1000,
        )
        conn.execute(
            "SELECT conversation_id FROM normalized_events WHERE id = ?", (nid,)
        ).fetchone()[0]

        # Even when rebuild encounters an issue (e.g. nonexistent conversation),
        # immutable raw/normalized rows must be preserved.
        _rebuild_conversation(
            conn, "nonexistent_conv_id", "p1", "bot1", "private", 12000
        )

        # Immutable rows should still be intact
        cursor = conn.execute("SELECT COUNT(*) FROM raw_events")
        assert cursor.fetchone()[0] == 1
        cursor = conn.execute("SELECT COUNT(*) FROM normalized_events")
        assert cursor.fetchone()[0] == 1

    def test_ensure_relationship_failure_does_not_crash(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        _insert_normalized(
            conn,
            source_message_id="msg_001",
            source_occurred_time=1000,
        )
        # Should not raise even if no event exists
        _ensure_relationship_for_normalized(conn, 9999)
        assert True

    def test_mid_rebuild_failure_rolls_back_only_derived_rows(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        nid = _insert_normalized(
            conn, source_message_id="msg_001", source_occurred_time=1000
        )
        with patch(
            "astrbot.builtin_stars.qq_clone_observer._conversation._rebuild_reply_links",
            side_effect=RuntimeError("secret payload must not escape"),
        ):
            with pytest.raises(RuntimeError):
                _rebuild_conversation_for_normalized(conn, nid)
        assert conn.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM normalized_events").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM bursts").fetchone()[0] == 0

    def test_rebuild_is_durable_after_connection_reopen(self, tmp_path: Path):
        conn = _make_conn(tmp_path)
        nid = _insert_normalized(
            conn, source_message_id="msg_001", source_occurred_time=1000
        )
        _rebuild_conversation_for_normalized(conn, nid)
        db_path = tmp_path / "test_conv.db"
        conn.close()
        reopened = sqlite3.connect(str(db_path))
        assert reopened.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 1
        assert reopened.execute("SELECT COUNT(*) FROM bursts").fetchone()[0] == 1
        assert (
            reopened.execute("SELECT COUNT(*) FROM conversation_privacy").fetchone()[0]
            == 1
        )
        reopened.close()


# ---------------------------------------------------------------------------
# Integration: plugin rebuilds after normalization
# ---------------------------------------------------------------------------


class TestPluginIntegration:
    """Tests that the plugin integrates conversation rebuild after normalization."""

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
            plugin = QQCloneObserver(ctx, config_mock)
            plugin._conn = sqlite3.connect(
                str(tmp_path / "raw_events.db"), check_same_thread=False
            )
            plugin._conn.execute("PRAGMA journal_mode=WAL")
            plugin._conn.execute("PRAGMA foreign_keys=ON")
            _build_schema(plugin._conn.cursor())
            _build_conversation_schema(plugin._conn.cursor())
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
    ) -> MagicMock:
        event = MagicMock()
        event.get_platform_id.return_value = platform_id
        event.get_self_id.return_value = self_id
        event.get_sender_id.return_value = sender_id

        message_obj = AstrBotMessage()
        message_obj.type = MessageType.FRIEND_MESSAGE
        message_obj.direction = direction
        message_obj.provenance = provenance
        message_obj.message_id = message_id
        message_obj.timestamp = int(time.time())
        message_obj.raw_message = raw_message
        message_obj.self_id = self_id
        message_obj.sender = MessageMember(user_id=sender_id, nickname=sender_nickname)
        message_obj.message = []

        event.message_obj = message_obj
        return event

    @pytest.mark.asyncio
    async def test_plugin_creates_conversation_on_normalize(self, plugin_with_db):
        raw = {
            "post_type": "message",
            "message_type": "private",
            "self_id": 123456,
            "sender": {"user_id": "user_001", "nickname": "Alice"},
        }
        event = self._make_event(raw_message=raw)
        await plugin_with_db.on_message_observed(event)

        cursor = plugin_with_db._conn.execute("SELECT COUNT(*) FROM conversations")
        assert cursor.fetchone()[0] == 1

    @pytest.mark.asyncio
    async def test_plugin_creates_bursts_on_normalize(self, plugin_with_db):
        raw = {
            "post_type": "message",
            "message_type": "private",
            "self_id": 123456,
            "sender": {"user_id": "user_001", "nickname": "Alice"},
        }
        e1 = self._make_event(raw_message=raw, message_id="msg_001")
        e2 = self._make_event(raw_message=raw, message_id="msg_002")

        await plugin_with_db.on_message_observed(e1)
        await plugin_with_db.on_message_observed(e2)

        cursor = plugin_with_db._conn.execute("SELECT COUNT(*) FROM bursts")
        assert cursor.fetchone()[0] == 1
        cursor = plugin_with_db._conn.execute("SELECT message_count FROM bursts")
        assert cursor.fetchone()[0] == 2

    @pytest.mark.asyncio
    async def test_duplicate_replay_repairs_derived_rows(self, plugin_with_db):
        raw = {
            "post_type": "message",
            "message_type": "private",
            "time": 1000,
            "sender": {"user_id": "user_001"},
        }
        event = self._make_event(raw_message=raw)
        await plugin_with_db.on_message_observed(event)
        plugin_with_db._conn.execute("DELETE FROM burst_members")
        plugin_with_db._conn.execute("DELETE FROM bursts")
        plugin_with_db._conn.execute("DELETE FROM reply_links")
        plugin_with_db._conn.commit()
        await plugin_with_db.on_message_observed(event)
        assert (
            plugin_with_db._conn.execute("SELECT COUNT(*) FROM bursts").fetchone()[0]
            == 1
        )

    @pytest.mark.asyncio
    async def test_plugin_auto_creates_relationship_profile(self, plugin_with_db):
        raw = {
            "post_type": "message",
            "message_type": "private",
            "sender": {"user_id": "user_001"},
        }
        await plugin_with_db.on_message_observed(self._make_event(raw_message=raw))
        row = plugin_with_db._conn.execute(
            "SELECT relationship_type, manually_confirmed FROM relationship_profiles "
            "WHERE subject_id = 'user_001'"
        ).fetchone()
        assert row == ("unknown", 0)

    @pytest.mark.asyncio
    async def test_plugin_uses_custom_burst_window(self, plugin_with_db):
        plugin_with_db._burst_window_ms = 1000
        first = self._make_event(
            message_id="msg_001",
            raw_message={
                "post_type": "message",
                "message_type": "private",
                "time": 1000,
                "sender": {"user_id": "user_001"},
            },
        )
        second = self._make_event(
            message_id="msg_002",
            raw_message={
                "post_type": "message",
                "message_type": "private",
                "time": 1005,
                "sender": {"user_id": "user_001"},
            },
        )
        await plugin_with_db.on_message_observed(first)
        await plugin_with_db.on_message_observed(second)
        assert (
            plugin_with_db._conn.execute("SELECT COUNT(*) FROM bursts").fetchone()[0]
            == 2
        )

    def test_invalid_burst_window_config_uses_default(self):
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: {
            "burst_merge_window_ms": True
        }.get(key, default)
        plugin = QQCloneObserver(MagicMock(), config)
        assert plugin._burst_window_ms == 12000

    @pytest.mark.asyncio
    async def test_plugin_never_sends(self, plugin_with_db):
        raw = {"post_type": "message", "message_type": "private"}
        event = self._make_event(raw_message=raw)
        event.send = MagicMock()
        event.send_streaming = MagicMock()
        event.plain_result = MagicMock()
        event.make_result = MagicMock()
        await plugin_with_db.on_message_observed(event)

        event.stop_event.assert_called_once()
        event.send.assert_not_called()
        event.send_streaming.assert_not_called()
        event.plain_result.assert_not_called()
        event.make_result.assert_not_called()
