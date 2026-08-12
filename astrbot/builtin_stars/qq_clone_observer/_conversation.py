"""Conversation builder for QQClone passive observer.

Builds deterministic, SQLite-backed conversations from committed immutable
``normalized_events`` facts. Delivers conversation assignment and ordering,
reply-link resolution, reproducible message bursts/semantic turns, editable
relationship profiles, and per-conversation privacy controls.

This module is observation-only and never sends messages or calls external
services. All SQLite work stays off the event loop.
"""

import hashlib
import json
import sqlite3
import time

_VALID_RELATIONSHIP_TYPES = frozenset(
    {
        "unknown",
        "stranger",
        "acquaintance",
        "friend",
        "close_friend",
        "family",
        "coworker",
        "client",
        "professional",
        "other",
    }
)

_VALID_PRIVACY_LEVELS = frozenset({"normal", "sensitive", "excluded"})

_DEFAULT_BURST_WINDOW_MS = 12000


def _build_conversation_schema(cursor: sqlite3.Cursor) -> None:
    """Create conversation, reply-link, burst, relationship, and privacy tables.

    All schema creation is idempotent. Foreign keys are expected to be enabled
    at the connection level.

    Args:
        cursor: SQLite cursor for the plugin database.
    """
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id     TEXT NOT NULL UNIQUE,
            platform_id         TEXT NOT NULL,
            self_id             TEXT NOT NULL,
            conversation_type   TEXT NOT NULL,
            burst_window_ms     INTEGER NOT NULL DEFAULT 12000,
            created_at          TEXT NOT NULL,
            updated_at          TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_platform_self
            ON conversations(platform_id, self_id);

        CREATE TABLE IF NOT EXISTS reply_links (
            id                              INTEGER PRIMARY KEY AUTOINCREMENT,
            normalized_event_id             INTEGER NOT NULL UNIQUE,
            conversation_id                 TEXT NOT NULL,
            source_reply_target_message_id  TEXT NOT NULL,
            resolved_normalized_event_id    INTEGER,
            FOREIGN KEY (normalized_event_id) REFERENCES normalized_events(id),
            FOREIGN KEY (resolved_normalized_event_id) REFERENCES normalized_events(id),
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
        );

        CREATE INDEX IF NOT EXISTS idx_reply_links_conv_id
            ON reply_links(conversation_id);

        CREATE INDEX IF NOT EXISTS idx_reply_links_resolved
            ON reply_links(source_reply_target_message_id, conversation_id);

        CREATE TABLE IF NOT EXISTS bursts (
            id                  TEXT PRIMARY KEY,
            conversation_id     TEXT NOT NULL,
            effective_sender_id TEXT NOT NULL,
            start_time_ms       INTEGER NOT NULL,
            end_time_ms         INTEGER NOT NULL,
            message_count       INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
        );

        CREATE INDEX IF NOT EXISTS idx_bursts_conv_id
            ON bursts(conversation_id);

        CREATE TABLE IF NOT EXISTS burst_members (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            burst_id            TEXT NOT NULL,
            normalized_event_id INTEGER NOT NULL UNIQUE,
            position            INTEGER NOT NULL,
            FOREIGN KEY (burst_id) REFERENCES bursts(id),
            FOREIGN KEY (normalized_event_id) REFERENCES normalized_events(id),
            UNIQUE(burst_id, position)
        );

        CREATE INDEX IF NOT EXISTS idx_burst_members_burst_id
            ON burst_members(burst_id);

        CREATE TABLE IF NOT EXISTS relationship_profiles (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            platform_id         TEXT NOT NULL,
            self_id             TEXT NOT NULL,
            subject_id          TEXT NOT NULL,
            relationship_type   TEXT NOT NULL DEFAULT 'unknown',
            closeness_score     REAL,
            notes               TEXT,
            manually_confirmed  INTEGER NOT NULL DEFAULT 0,
            updated_at          TEXT NOT NULL,
            UNIQUE(platform_id, self_id, subject_id)
        );

        CREATE INDEX IF NOT EXISTS idx_relationship_profiles_subject
            ON relationship_profiles(platform_id, self_id, subject_id);

        CREATE TABLE IF NOT EXISTS conversation_privacy (
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id             TEXT NOT NULL UNIQUE,
            observe_enabled             INTEGER NOT NULL DEFAULT 1,
            persona_learning_enabled    INTEGER NOT NULL DEFAULT 1,
            training_export_enabled     INTEGER NOT NULL DEFAULT 0,
            clone_reply_enabled         INTEGER NOT NULL DEFAULT 0,
            privacy_level               TEXT NOT NULL DEFAULT 'normal',
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
        );
    """)


def _get_or_create_conversation(
    conn: sqlite3.Connection,
    conversation_id: str,
    platform_id: str,
    self_id: str,
    conversation_type: str,
    burst_window_ms: int = _DEFAULT_BURST_WINDOW_MS,
) -> dict:
    """Get an existing conversation row or create a new one.

    Args:
        conn: SQLite connection.
        conversation_id: Stable conversation ID string.
        platform_id: Platform instance ID.
        self_id: Bot self ID.
        conversation_type: ``private``, ``group``, or ``unknown``.

    Returns:
        Dict with conversation row fields.
    """
    cursor = conn.execute(
        "SELECT id, conversation_id, platform_id, self_id, conversation_type, "
        "burst_window_ms, created_at, updated_at FROM conversations "
        "WHERE conversation_id = ?",
        (conversation_id,),
    )
    row = cursor.fetchone()
    if row is not None:
        if row[5] != burst_window_ms:
            conn.execute(
                "UPDATE conversations SET burst_window_ms = ? WHERE conversation_id = ?",
                (burst_window_ms, conversation_id),
            )
        return {
            "id": row[0],
            "conversation_id": row[1],
            "platform_id": row[2],
            "self_id": row[3],
            "conversation_type": row[4],
            "burst_window_ms": burst_window_ms,
            "created_at": row[6],
            "updated_at": row[7],
        }

    now = str(_current_time_ms())
    conn.execute(
        "INSERT INTO conversations "
        "(conversation_id, platform_id, self_id, conversation_type, burst_window_ms, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            conversation_id,
            platform_id,
            self_id,
            conversation_type,
            burst_window_ms,
            now,
            now,
        ),
    )
    cursor = conn.execute(
        "SELECT id, conversation_id, platform_id, self_id, conversation_type, "
        "burst_window_ms, created_at, updated_at FROM conversations "
        "WHERE conversation_id = ?",
        (conversation_id,),
    )
    row = cursor.fetchone()
    return {
        "id": row[0],
        "conversation_id": row[1],
        "platform_id": row[2],
        "self_id": row[3],
        "conversation_type": row[4],
        "burst_window_ms": row[5],
        "created_at": row[6],
        "updated_at": row[7],
    }


def _current_time_ms() -> int:
    """Return the current Unix time in milliseconds.

    Returns:
        Current Unix time in milliseconds.
    """
    return int(time.time() * 1000)


def _get_conversation_events(
    conn: sqlite3.Connection,
    conversation_id: str,
) -> list[dict]:
    """Get all normalized events for a conversation in deterministic order.

    Sort key (documented):
    1. ``source_occurred_time`` (nullable — COALESCE to 0 when absent).
    2. ``normalized_processing_time`` (stored as TEXT, compared as integer ms).
    3. ``id`` (stable row identity as final tiebreaker).

    Args:
        conn: SQLite connection.
        conversation_id: Stable conversation ID.

    Returns:
        List of normalized event dicts in deterministic order.
    """
    cursor = conn.execute(
        "SELECT normalized_events.id, normalized_events.raw_event_id, "
        "normalized_events.platform_id, normalized_events.self_id, "
        "normalized_events.conversation_id, "
        "normalized_events.conversation_type, normalized_events.sender_id, "
        "normalized_events.sender_display_name, normalized_events.direction, "
        "normalized_events.source_occurred_time, "
        "normalized_events.normalized_processing_time, "
        "normalized_events.reply_target_message_id, "
        "normalized_events.source_message_id, raw_events.stable_fingerprint "
        "FROM normalized_events "
        "JOIN raw_events ON raw_events.id = normalized_events.raw_event_id "
        "WHERE normalized_events.conversation_id = ? "
        "ORDER BY "
        "  COALESCE(normalized_events.source_occurred_time, "
        "    CAST(COALESCE(NULLIF(normalized_events.normalized_processing_time, ''), "
        "      '0') AS INTEGER)) ASC, "
        "  CAST(COALESCE(NULLIF(normalized_events.normalized_processing_time, ''), "
        "    '0') AS INTEGER) ASC, "
        "  normalized_events.id ASC",
        (conversation_id,),
    )
    columns = [
        "id",
        "raw_event_id",
        "platform_id",
        "self_id",
        "conversation_id",
        "conversation_type",
        "sender_id",
        "sender_display_name",
        "direction",
        "source_occurred_time",
        "normalized_processing_time",
        "reply_target_message_id",
        "source_message_id",
        "stable_fingerprint",
    ]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _event_time_ms(event: dict) -> int:
    """Get deterministic event time in milliseconds.

    Multiplies ``source_occurred_time`` by 1000 because OneBot timestamps
    currently normalize as seconds. Falls back to ``normalized_processing_time``
    (also in seconds) when source time is absent, also multiplied by 1000.

    Args:
        event: Normalized event dict with at least ``source_occurred_time``
            and ``normalized_processing_time``.

    Returns:
        Event time in milliseconds.
    """
    src = event.get("source_occurred_time")
    if src is not None:
        return src * 1000
    proc = event.get("normalized_processing_time") or "0"
    try:
        return int(proc) * 1000
    except (ValueError, TypeError):
        return 0


def _effective_sender(event: dict, self_id: str) -> str:
    """Determine the effective sender for a normalized event.

    For outgoing messages with no sender evidence, falls back to ``self_id``.
    Otherwise uses the normalized ``sender_id`` or ``"unknown"``.

    Args:
        event: Normalized event dict.
        self_id: Bot self ID for the account.

    Returns:
        Effective sender identifier string.
    """
    direction = event.get("direction", "unknown")
    if direction == "outgoing":
        return self_id
    sender = event.get("sender_id")
    if sender:
        return sender
    return "unknown"


def _rebuild_conversation(
    conn: sqlite3.Connection,
    conversation_id: str,
    platform_id: str,
    self_id: str,
    conversation_type: str,
    burst_window_ms: int = _DEFAULT_BURST_WINDOW_MS,
) -> None:
    """Rebuild all derived rows (bursts, reply links) for a conversation.

    Must be called in the same transaction scope as the normalisation that
    triggered it. Derived rows are replaced transactionally; raw and normalized
    rows remain immutable.

    Args:
        conn: SQLite connection.
        conversation_id: Stable conversation ID.
        platform_id: Platform instance ID.
        self_id: Bot self ID.
        conversation_type: Conversation type string.
        burst_window_ms: Maximum gap in milliseconds for same-sender bursts.
    """
    # Ensure conversation row exists
    _get_or_create_conversation(
        conn,
        conversation_id,
        platform_id,
        self_id,
        conversation_type,
        burst_window_ms,
    )

    # Ensure a privacy row exists (created with defaults if missing)
    _ensure_privacy_row(conn, conversation_id)

    # Check privacy: skip rebuild if excluded or observation disabled
    privacy = _get_conversation_privacy(conn, conversation_id)
    if privacy is not None:
        if privacy["privacy_level"] == "excluded" or not privacy["observe_enabled"]:
            _clear_derived_rows(conn, conversation_id)
            return

    events = _get_conversation_events(conn, conversation_id)
    if not events:
        _clear_derived_rows(conn, conversation_id)
        return

    # Clear old derived rows for this conversation
    _clear_derived_rows(conn, conversation_id)

    # Rebuild bursts
    _rebuild_bursts(conn, conversation_id, events, self_id, burst_window_ms)

    # Rebuild reply links
    _rebuild_reply_links(conn, conversation_id, events)


def _clear_derived_rows(conn: sqlite3.Connection, conversation_id: str) -> None:
    """Delete all derived rows (bursts, burst_members, reply_links) for a conversation.

    Args:
        conn: SQLite connection.
        conversation_id: Stable conversation ID.
    """
    conn.execute(
        "DELETE FROM burst_members WHERE burst_id IN "
        "(SELECT id FROM bursts WHERE conversation_id = ?)",
        (conversation_id,),
    )
    conn.execute("DELETE FROM bursts WHERE conversation_id = ?", (conversation_id,))
    conn.execute(
        "DELETE FROM reply_links WHERE conversation_id = ?", (conversation_id,)
    )


def _rebuild_bursts(
    conn: sqlite3.Connection,
    conversation_id: str,
    events: list[dict],
    self_id: str,
    burst_window_ms: int,
) -> None:
    """Rebuild burst rows for a list of ordered events.

    A burst is a consecutive sequence in deterministic conversation order with
    the same effective sender and an event-time gap no greater than the window.
    Any intervening different sender starts a new burst.

    Args:
        conn: SQLite connection.
        conversation_id: Stable conversation ID.
        events: Ordered list of normalized event dicts.
        self_id: Bot self ID for effective sender fallback.
        burst_window_ms: Maximum gap in milliseconds.
    """
    if not events:
        return

    current_sender = _effective_sender(events[0], self_id)
    current_start = _event_time_ms(events[0])
    current_end = current_start
    current_ids = [
        (
            events[0]["id"],
            0,
            events[0]["source_message_id"] or events[0]["stable_fingerprint"],
        )
    ]
    position = 0

    for event in events[1:]:
        sender = _effective_sender(event, self_id)
        event_time = _event_time_ms(event)
        gap = event_time - current_end

        if sender == current_sender and gap <= burst_window_ms:
            # Same burst
            current_end = event_time
            position += 1
            current_ids.append(
                (
                    event["id"],
                    position,
                    event["source_message_id"] or event["stable_fingerprint"],
                )
            )
        else:
            # Flush current burst
            _insert_burst(
                conn,
                conversation_id,
                current_sender,
                current_start,
                current_end,
                current_ids,
            )
            # Start new burst
            current_sender = sender
            current_start = event_time
            current_end = event_time
            position = 0
            current_ids = [
                (
                    event["id"],
                    position,
                    event["source_message_id"] or event["stable_fingerprint"],
                )
            ]

    # Flush last burst
    _insert_burst(
        conn, conversation_id, current_sender, current_start, current_end, current_ids
    )


def _insert_burst(
    conn: sqlite3.Connection,
    conversation_id: str,
    sender: str,
    start_time_ms: int,
    end_time_ms: int,
    member_ids: list[tuple[int, int, str]],
) -> None:
    """Insert a burst row and its membership rows.

    Args:
        conn: SQLite connection.
        conversation_id: Stable conversation ID.
        sender: Effective sender ID.
        start_time_ms: Burst start time in milliseconds.
        end_time_ms: Burst end time in milliseconds.
        member_ids: List of normalized event ID, position, and stable raw
            fingerprint tuples.
    """
    burst_key = json.dumps(
        [conversation_id, [fingerprint for _, _, fingerprint in member_ids]],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    burst_id = hashlib.sha256(burst_key.encode("utf-8")).hexdigest()
    cursor = conn.execute(
        "INSERT INTO bursts "
        "(id, conversation_id, effective_sender_id, start_time_ms, end_time_ms, "
        "message_count) VALUES (?, ?, ?, ?, ?, ?)",
        (
            burst_id,
            conversation_id,
            sender,
            start_time_ms,
            end_time_ms,
            len(member_ids),
        ),
    )
    assert cursor.rowcount == 1
    for nid, pos, _ in member_ids:
        conn.execute(
            "INSERT OR IGNORE INTO burst_members "
            "(burst_id, normalized_event_id, position) VALUES (?, ?, ?)",
            (burst_id, nid, pos),
        )


def _rebuild_reply_links(
    conn: sqlite3.Connection,
    conversation_id: str,
    events: list[dict],
) -> None:
    """Rebuild reply link rows for a list of events.

    Resolves the target by ``source_message_id`` only inside the same
    conversation. Preserves the source reply target ID even when no local
    target exists.

    Args:
        conn: SQLite connection.
        conversation_id: Stable conversation ID.
        events: Ordered list of normalized event dicts.
    """
    for event in events:
        reply_target = event.get("reply_target_message_id")
        if not reply_target:
            continue

        # Try to resolve within the same conversation
        resolved_id = None
        cursor = conn.execute(
            "SELECT id FROM normalized_events "
            "WHERE conversation_id = ? AND source_message_id = ? LIMIT 1",
            (conversation_id, reply_target),
        )
        row = cursor.fetchone()
        if row is not None:
            resolved_id = row[0]

        conn.execute(
            "INSERT OR IGNORE INTO reply_links "
            "(normalized_event_id, conversation_id, source_reply_target_message_id, "
            "resolved_normalized_event_id) "
            "VALUES (?, ?, ?, ?)",
            (event["id"], conversation_id, reply_target, resolved_id),
        )


def _rebuild_conversation_for_normalized(
    conn: sqlite3.Connection,
    normalized_event_id: int,
    burst_window_ms: int = _DEFAULT_BURST_WINDOW_MS,
) -> None:
    """Rebuild the conversation for a single normalized event.

    Fetches the event's conversation metadata and triggers a full rebuild of
    the affected conversation.

    Args:
        conn: SQLite connection.
        normalized_event_id: Normalized event row ID.
        burst_window_ms: Maximum gap in milliseconds for same-sender bursts.
    """
    conn.execute("SAVEPOINT qqclone_conversation_rebuild")
    try:
        cursor = conn.execute(
            "SELECT conversation_id, platform_id, self_id, conversation_type "
            "FROM normalized_events WHERE id = ?",
            (normalized_event_id,),
        )
        row = cursor.fetchone()
        if row is not None:
            _ensure_relationship_for_normalized(conn, normalized_event_id)
            _rebuild_conversation(
                conn,
                row[0],
                row[1],
                row[2],
                row[3],
                burst_window_ms,
            )
        conn.execute("RELEASE SAVEPOINT qqclone_conversation_rebuild")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT qqclone_conversation_rebuild")
        conn.execute("RELEASE SAVEPOINT qqclone_conversation_rebuild")
        raise


# ---------------------------------------------------------------------------
# Relationship profile API
# ---------------------------------------------------------------------------


def _create_relationship_profile(
    conn: sqlite3.Connection,
    platform_id: str,
    self_id: str,
    subject_id: str,
    relationship_type: str = "unknown",
    closeness_score: float | None = None,
    notes: str | None = None,
    manually_confirmed: bool = False,
) -> dict:
    """Create a new relationship profile, ignoring duplicates.

    Args:
        conn: SQLite connection.
        platform_id: Platform instance ID.
        self_id: Bot self ID.
        subject_id: Subject identifier (QQ number, user ID).
        relationship_type: Valid relationship type string.
        closeness_score: Optional closeness score (0.0–1.0).
        notes: Optional free-text notes.
        manually_confirmed: Whether this profile was manually confirmed.

    Returns:
        Dict with the profile row fields (existing if already present).
    """
    if relationship_type not in _VALID_RELATIONSHIP_TYPES:
        raise ValueError(f"invalid relationship_type: {relationship_type}")
    if closeness_score is not None and not (0.0 <= closeness_score <= 1.0):
        raise ValueError(
            f"closeness_score must be in [0.0, 1.0], got {closeness_score}"
        )

    now = str(_current_time_ms())
    conn.execute("SAVEPOINT qqclone_relationship_create")
    try:
        conn.execute(
            "INSERT OR IGNORE INTO relationship_profiles "
            "(platform_id, self_id, subject_id, relationship_type, closeness_score, "
            "notes, manually_confirmed, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                platform_id,
                self_id,
                subject_id,
                relationship_type,
                closeness_score,
                notes,
                1 if manually_confirmed else 0,
                now,
            ),
        )
        conn.execute("RELEASE SAVEPOINT qqclone_relationship_create")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT qqclone_relationship_create")
        conn.execute("RELEASE SAVEPOINT qqclone_relationship_create")
        raise
    result = _get_relationship_profile(conn, platform_id, self_id, subject_id)
    assert result is not None  # just inserted or already existed
    return result


def _get_relationship_profile(
    conn: sqlite3.Connection,
    platform_id: str,
    self_id: str,
    subject_id: str,
) -> dict | None:
    """Get a relationship profile by account and subject scope.

    Args:
        conn: SQLite connection.
        platform_id: Platform instance ID.
        self_id: Bot self ID.
        subject_id: Subject identifier.

    Returns:
        Profile dict or None when not found.
    """
    cursor = conn.execute(
        "SELECT id, platform_id, self_id, subject_id, relationship_type, "
        "closeness_score, notes, manually_confirmed, updated_at "
        "FROM relationship_profiles "
        "WHERE platform_id = ? AND self_id = ? AND subject_id = ?",
        (platform_id, self_id, subject_id),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "platform_id": row[1],
        "self_id": row[2],
        "subject_id": row[3],
        "relationship_type": row[4],
        "closeness_score": row[5],
        "notes": row[6],
        "manually_confirmed": bool(row[7]),
        "updated_at": row[8],
    }


def _update_relationship_profile(
    conn: sqlite3.Connection,
    platform_id: str,
    self_id: str,
    subject_id: str,
    relationship_type: str | None = None,
    closeness_score: float | None = None,
    notes: str | None = None,
    manually_confirmed: bool | None = None,
) -> dict:
    """Update an existing relationship profile.

    Args:
        conn: SQLite connection.
        platform_id: Platform instance ID.
        self_id: Bot self ID.
        subject_id: Subject identifier.
        relationship_type: Optional new relationship type.
        closeness_score: Optional new closeness score (0.0–1.0).
        notes: Optional new notes text.
        manually_confirmed: Optional new manual confirmation state.

    Returns:
        Updated profile dict.

    Raises:
        ValueError: If the profile does not exist, or if ``relationship_type``
            is not valid, or if ``closeness_score`` is outside 0.0–1.0.
    """
    current = _get_relationship_profile(conn, platform_id, self_id, subject_id)
    if current is None:
        raise ValueError("relationship profile not found")

    new_type = (
        relationship_type
        if relationship_type is not None
        else current["relationship_type"]
    )
    if new_type not in _VALID_RELATIONSHIP_TYPES:
        raise ValueError(f"invalid relationship_type: {new_type}")

    new_score = (
        closeness_score if closeness_score is not None else current["closeness_score"]
    )
    if new_score is not None and not (0.0 <= new_score <= 1.0):
        raise ValueError(f"closeness_score must be in [0.0, 1.0], got {new_score}")

    new_notes = notes if notes is not None else current["notes"]
    new_confirmed = (
        manually_confirmed
        if manually_confirmed is not None
        else current["manually_confirmed"]
    )
    now = str(_current_time_ms())

    conn.execute("SAVEPOINT qqclone_relationship_update")
    try:
        conn.execute(
            "UPDATE relationship_profiles SET "
            "relationship_type = ?, closeness_score = ?, notes = ?, "
            "manually_confirmed = ?, updated_at = ? "
            "WHERE platform_id = ? AND self_id = ? AND subject_id = ?",
            (
                new_type,
                new_score,
                new_notes,
                1 if new_confirmed else 0,
                now,
                platform_id,
                self_id,
                subject_id,
            ),
        )
        conn.execute("RELEASE SAVEPOINT qqclone_relationship_update")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT qqclone_relationship_update")
        conn.execute("RELEASE SAVEPOINT qqclone_relationship_update")
        raise

    result = _get_relationship_profile(conn, platform_id, self_id, subject_id)
    assert result is not None
    return result


def _ensure_relationship_for_normalized(
    conn: sqlite3.Connection,
    normalized_event_id: int,
) -> None:
    """Auto-create a relationship profile for a non-self sender in a normalized event.

    For private conversations, the peer is the subject. For group conversations,
    non-self message senders get a profile. Does not overwrite existing profiles.

    Args:
        conn: SQLite connection.
        normalized_event_id: Normalized event row ID.
    """
    row = conn.execute(
        "SELECT normalized_events.platform_id, normalized_events.self_id, "
        "normalized_events.conversation_type, normalized_events.sender_id, "
        "normalized_events.direction, raw_events.payload_json "
        "FROM normalized_events "
        "JOIN raw_events ON raw_events.id = normalized_events.raw_event_id "
        "WHERE normalized_events.id = ?",
        (normalized_event_id,),
    ).fetchone()
    if row is None:
        return

    platform_id, self_id, conversation_type, subject_id, direction, payload_json = row

    if conversation_type == "private" and (
        not subject_id or subject_id == self_id or direction == "outgoing"
    ):
        try:
            payload = json.loads(payload_json)
        except (json.JSONDecodeError, TypeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        raw_subject = (
            payload.get("target_id")
            if direction == "outgoing"
            else payload.get("user_id")
        )
        subject_id = str(raw_subject) if raw_subject is not None else None

    if (
        conversation_type not in {"private", "group"}
        or not subject_id
        or subject_id == self_id
    ):
        return
    if _get_relationship_profile(conn, platform_id, self_id, subject_id) is None:
        _create_relationship_profile(
            conn,
            platform_id,
            self_id,
            subject_id,
            relationship_type="unknown",
        )


# ---------------------------------------------------------------------------
# Privacy control API
# ---------------------------------------------------------------------------


def _ensure_privacy_row(
    conn: sqlite3.Connection,
    conversation_id: str,
) -> dict:
    """Ensure a privacy control row exists for a conversation, creating with defaults if needed.

    Args:
        conn: SQLite connection.
        conversation_id: Stable conversation ID.

    Returns:
        Privacy control dict.
    """
    privacy = _get_conversation_privacy(conn, conversation_id)
    if privacy is not None:
        return privacy

    conn.execute(
        "INSERT OR IGNORE INTO conversation_privacy (conversation_id) VALUES (?)",
        (conversation_id,),
    )
    result = _get_conversation_privacy(conn, conversation_id)
    assert result is not None
    return result


def _get_conversation_privacy(
    conn: sqlite3.Connection,
    conversation_id: str,
) -> dict | None:
    """Get privacy controls for a conversation.

    Args:
        conn: SQLite connection.
        conversation_id: Stable conversation ID.

    Returns:
        Privacy control dict or None when not found.
    """
    cursor = conn.execute(
        "SELECT id, conversation_id, observe_enabled, persona_learning_enabled, "
        "training_export_enabled, clone_reply_enabled, privacy_level "
        "FROM conversation_privacy WHERE conversation_id = ?",
        (conversation_id,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "conversation_id": row[1],
        "observe_enabled": bool(row[2]),
        "persona_learning_enabled": bool(row[3]),
        "training_export_enabled": bool(row[4]),
        "clone_reply_enabled": bool(row[5]),
        "privacy_level": row[6],
    }


def _update_conversation_privacy(
    conn: sqlite3.Connection,
    conversation_id: str,
    observe_enabled: bool | None = None,
    persona_learning_enabled: bool | None = None,
    training_export_enabled: bool | None = None,
    clone_reply_enabled: bool | None = None,
    privacy_level: str | None = None,
) -> dict:
    """Update privacy controls for a conversation.

    Args:
        conn: SQLite connection.
        conversation_id: Stable conversation ID.
        observe_enabled: Whether observation is enabled.
        persona_learning_enabled: Whether persona learning is enabled.
        training_export_enabled: Whether training export is enabled.
        clone_reply_enabled: Whether clone replies are enabled.
        privacy_level: Privacy level (``normal``, ``sensitive``, ``excluded``).

    Returns:
        Updated privacy control dict.

    Raises:
        ValueError: If the privacy row does not exist, or if ``privacy_level``
            is not valid.
    """
    boolean_values = {
        "observe_enabled": observe_enabled,
        "persona_learning_enabled": persona_learning_enabled,
        "training_export_enabled": training_export_enabled,
        "clone_reply_enabled": clone_reply_enabled,
    }
    for name, value in boolean_values.items():
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{name} must be a bool")

    current = _get_conversation_privacy(conn, conversation_id)
    if current is None:
        raise ValueError("conversation_privacy row not found")

    new_level = privacy_level if privacy_level is not None else current["privacy_level"]
    if new_level not in _VALID_PRIVACY_LEVELS:
        raise ValueError(f"invalid privacy_level: {new_level}")

    new_observe = (
        observe_enabled if observe_enabled is not None else current["observe_enabled"]
    )
    new_persona = (
        persona_learning_enabled
        if persona_learning_enabled is not None
        else current["persona_learning_enabled"]
    )
    new_export = (
        training_export_enabled
        if training_export_enabled is not None
        else current["training_export_enabled"]
    )
    new_clone = (
        clone_reply_enabled
        if clone_reply_enabled is not None
        else current["clone_reply_enabled"]
    )

    conn.execute("SAVEPOINT qqclone_privacy_update")
    try:
        conn.execute(
            "UPDATE conversation_privacy SET "
            "observe_enabled = ?, persona_learning_enabled = ?, "
            "training_export_enabled = ?, clone_reply_enabled = ?, "
            "privacy_level = ? "
            "WHERE conversation_id = ?",
            (
                1 if new_observe else 0,
                1 if new_persona else 0,
                1 if new_export else 0,
                1 if new_clone else 0,
                new_level,
                conversation_id,
            ),
        )
        conversation = conn.execute(
            "SELECT platform_id, self_id, conversation_type, burst_window_ms "
            "FROM conversations "
            "WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if conversation is not None:
            _rebuild_conversation(
                conn,
                conversation_id,
                conversation[0],
                conversation[1],
                conversation[2],
                conversation[3],
            )
        conn.execute("RELEASE SAVEPOINT qqclone_privacy_update")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT qqclone_privacy_update")
        conn.execute("RELEASE SAVEPOINT qqclone_privacy_update")
        raise

    result = _get_conversation_privacy(conn, conversation_id)
    assert result is not None
    return result
