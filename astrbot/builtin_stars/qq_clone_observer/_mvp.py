"""Small local-first learning and shadow-draft layer for QQClone.

The module intentionally keeps the MVP narrow: verified source-account replies
become local examples, a deterministic style summary is maintained, and clone
incoming messages can produce reviewable drafts. Nothing in this module sends a
message or performs an outbound platform action.
"""

import json
import re
import sqlite3
import time

_REVIEW_STATUSES = frozenset({"approved", "edited", "rejected"})


def _build_mvp_schema(cursor: sqlite3.Cursor) -> None:
    """Create the human-example, persona, and shadow-draft tables.

    Args:
        cursor: SQLite cursor for the plugin database.
    """
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS human_examples (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            normalized_event_id INTEGER NOT NULL UNIQUE,
            platform_id         TEXT NOT NULL,
            self_id             TEXT NOT NULL,
            conversation_id     TEXT NOT NULL,
            conversation_type   TEXT NOT NULL,
            subject_id          TEXT,
            group_id            TEXT,
            response_text       TEXT NOT NULL,
            context_json        TEXT NOT NULL,
            created_at          INTEGER NOT NULL,
            FOREIGN KEY (normalized_event_id) REFERENCES normalized_events(id)
        );

        CREATE INDEX IF NOT EXISTS idx_human_examples_source
            ON human_examples(platform_id, self_id, id DESC);

        CREATE TABLE IF NOT EXISTS shadow_drafts (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            normalized_event_id     INTEGER NOT NULL UNIQUE,
            conversation_id         TEXT NOT NULL,
            source_platform_id      TEXT NOT NULL,
            source_self_id          TEXT NOT NULL,
            provider_id             TEXT,
            model_name              TEXT,
            input_json              TEXT NOT NULL,
            draft_text              TEXT,
            status                  TEXT NOT NULL DEFAULT 'pending',
            edited_text             TEXT,
            review_note             TEXT,
            error_type              TEXT,
            created_at              INTEGER NOT NULL,
            reviewed_at             INTEGER,
            FOREIGN KEY (normalized_event_id) REFERENCES normalized_events(id)
        );

        CREATE INDEX IF NOT EXISTS idx_shadow_drafts_status
            ON shadow_drafts(status, id DESC);
    """)


def _segments_to_text(segments_json: str) -> str:
    """Convert normalized message segments to safe prompt text.

    Args:
        segments_json: JSON encoded normalized segment list.

    Returns:
        Redacted plain text with small placeholders for non-text segments.
    """
    try:
        segments = json.loads(segments_json)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(segments, list):
        return ""

    parts: list[str] = []
    labels = {
        "image": "[图片]",
        "record": "[语音]",
        "audio": "[语音]",
        "video": "[视频]",
        "file": "[文件]",
        "face": "[表情]",
    }
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        kind = segment.get("kind")
        data = segment.get("data")
        if kind == "text" and isinstance(data, dict):
            text = data.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif kind in labels:
            parts.append(labels[kind])

    text = "".join(parts).strip()
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[邮箱]", text)
    text = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[手机号]", text)
    return text[:1000]


def _capture_human_example(
    conn: sqlite3.Connection,
    normalized_event_id: int,
    context_limit: int = 8,
) -> bool:
    """Store an eligible source-human reply and refresh its style summary.

    Args:
        conn: SQLite connection.
        normalized_event_id: Normalized event being considered.
        context_limit: Maximum preceding messages stored with the example.

    Returns:
        Whether a new example was stored.
    """
    row = conn.execute(
        "SELECT n.platform_id, n.self_id, n.conversation_id, "
        "n.conversation_type, n.sender_id, n.group_id, "
        "n.normalized_segments_json, r.payload_json, "
        "p.observe_enabled, p.persona_learning_enabled, p.privacy_level "
        "FROM normalized_events n "
        "JOIN raw_events r ON r.id = n.raw_event_id "
        "JOIN conversation_privacy p ON p.conversation_id = n.conversation_id "
        "WHERE n.id = ? AND n.account_role = 'source' "
        "AND n.direction = 'outgoing' AND n.actor_provenance = 'human'",
        (normalized_event_id,),
    ).fetchone()
    if row is None or not row[8] or not row[9] or row[10] == "excluded":
        return False

    response_text = _segments_to_text(row[6])
    if not response_text or response_text.startswith("/"):
        return False

    subject_id = row[4]
    if row[3] == "private":
        try:
            payload = json.loads(row[7])
        except (json.JSONDecodeError, TypeError):
            payload = {}
        if isinstance(payload, dict):
            raw_subject = payload.get("user_id") or payload.get("target_id")
            if raw_subject is not None:
                subject_id = str(raw_subject)

    preceding = conn.execute(
        "SELECT direction, sender_id, normalized_segments_json "
        "FROM normalized_events WHERE conversation_id = ? AND id < ? "
        "ORDER BY id DESC LIMIT ?",
        (row[2], normalized_event_id, max(1, context_limit)),
    ).fetchall()
    context = []
    for direction, sender_id, segments_json in reversed(preceding):
        text = _segments_to_text(segments_json)
        if text:
            context.append(
                {
                    "role": "self" if direction == "outgoing" else "peer",
                    "text": text,
                }
            )

    cursor = conn.execute(
        "INSERT OR IGNORE INTO human_examples "
        "(normalized_event_id, platform_id, self_id, conversation_id, "
        "conversation_type, subject_id, group_id, response_text, context_json, "
        "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            normalized_event_id,
            row[0],
            row[1],
            row[2],
            row[3],
            subject_id,
            row[5],
            response_text,
            json.dumps(context, ensure_ascii=False),
            int(time.time() * 1000),
        ),
    )
    if cursor.rowcount != 1:
        return False

    conn.commit()
    return True


def _prepare_shadow_request(
    conn: sqlite3.Connection,
    normalized_event_id: int,
    source_platform_id: str,
    source_self_id: str,
    context_limit: int = 8,
    example_limit: int = 5,
) -> dict | None:
    """Build a small local prompt payload for an eligible clone message.

    Args:
        conn: SQLite connection.
        normalized_event_id: Incoming clone normalized event ID.
        source_platform_id: Platform ID of the source account to imitate.
        source_self_id: Self ID of the source account to imitate.
        context_limit: Maximum recent clone-conversation messages.
        example_limit: Maximum verified human examples.

    Returns:
        Prompt data, or None when the event is ineligible or already drafted.
    """
    if conn.execute(
        "SELECT 1 FROM shadow_drafts WHERE normalized_event_id = ?",
        (normalized_event_id,),
    ).fetchone():
        return None

    row = conn.execute(
        "SELECT n.conversation_id, n.conversation_type, n.sender_id, n.group_id, "
        "n.normalized_segments_json, p.observe_enabled, p.privacy_level "
        "FROM normalized_events n "
        "JOIN conversation_privacy p ON p.conversation_id = n.conversation_id "
        "WHERE n.id = ? AND n.account_role = 'clone' AND n.direction = 'incoming'",
        (normalized_event_id,),
    ).fetchone()
    if row is None or not row[5] or row[6] == "excluded":
        return None

    incoming_text = _segments_to_text(row[4])
    if not incoming_text or incoming_text.startswith("/qqclone_"):
        return None

    recent_rows = conn.execute(
        "SELECT direction, sender_id, normalized_segments_json "
        "FROM normalized_events WHERE conversation_id = ? AND id <= ? "
        "ORDER BY id DESC LIMIT ?",
        (row[0], normalized_event_id, max(1, context_limit)),
    ).fetchall()
    recent_messages = []
    for direction, sender_id, segments_json in reversed(recent_rows):
        text = _segments_to_text(segments_json)
        if text:
            recent_messages.append(
                {
                    "role": "self" if direction == "outgoing" else "peer",
                    "text": text,
                }
            )

    examples = conn.execute(
        "SELECT h.id, h.response_text, h.context_json FROM human_examples h "
        "JOIN conversation_privacy p ON p.conversation_id = h.conversation_id "
        "WHERE h.platform_id = ? AND h.self_id = ? AND p.observe_enabled = 1 "
        "AND p.persona_learning_enabled = 1 AND p.privacy_level != 'excluded' "
        "ORDER BY CASE "
        "WHEN (? = 'private' AND h.subject_id = ?) THEN 0 "
        "WHEN (? = 'group' AND h.group_id = ?) THEN 0 ELSE 1 END, h.id DESC LIMIT ?",
        (
            source_platform_id,
            source_self_id,
            row[1],
            row[2],
            row[1],
            row[3],
            max(1, example_limit),
        ),
    ).fetchall()
    persona = _get_persona_profile(conn, source_platform_id)
    return {
        "normalized_event_id": normalized_event_id,
        "conversation_id": row[0],
        "incoming_text": incoming_text,
        "recent_messages": recent_messages,
        "examples": [
            {
                "id": item[0],
                "response": item[1],
                "context": json.loads(item[2]),
            }
            for item in examples
        ],
        "persona_summary": (
            persona["summary_text"] if persona else "暂无已验证的风格样本。"
        ),
    }


def _store_shadow_draft(
    conn: sqlite3.Connection,
    request: dict,
    source_platform_id: str,
    source_self_id: str,
    provider_id: str | None,
    model_name: str | None,
    draft_text: str | None,
    error_type: str | None = None,
) -> int:
    """Persist one idempotent shadow generation result.

    Args:
        conn: SQLite connection.
        request: Local prompt payload returned by ``_prepare_shadow_request``.
        source_platform_id: Source account platform ID.
        source_self_id: Source account self ID.
        provider_id: AstrBot provider ID, when available.
        model_name: Provider model name, when available.
        draft_text: Generated text, or None on failure.
        error_type: Exception class or stable failure label.

    Returns:
        Stored draft row ID.
    """
    cursor = conn.execute(
        "INSERT OR IGNORE INTO shadow_drafts "
        "(normalized_event_id, conversation_id, source_platform_id, "
        "source_self_id, provider_id, model_name, input_json, draft_text, "
        "status, error_type, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            request["normalized_event_id"],
            request["conversation_id"],
            source_platform_id,
            source_self_id,
            provider_id,
            model_name,
            json.dumps(request, ensure_ascii=False),
            draft_text,
            "pending" if draft_text else "failed",
            error_type,
            int(time.time() * 1000),
        ),
    )
    if cursor.rowcount == 1:
        draft_id = int(cursor.lastrowid)
    else:
        row = conn.execute(
            "SELECT id FROM shadow_drafts WHERE normalized_event_id = ?",
            (request["normalized_event_id"],),
        ).fetchone()
        if row is None:
            raise ValueError("shadow draft was not stored")
        draft_id = int(row[0])
    conn.commit()
    return draft_id


def _list_shadow_drafts(conn: sqlite3.Connection, limit: int = 5) -> list[dict]:
    """Return recent reviewable drafts.

    Args:
        conn: SQLite connection.
        limit: Maximum result count.

    Returns:
        Draft dictionaries ordered newest first.
    """
    rows = conn.execute(
        "SELECT id, conversation_id, draft_text, status, edited_text, error_type, "
        "created_at FROM shadow_drafts ORDER BY id DESC LIMIT ?",
        (min(max(limit, 1), 20),),
    ).fetchall()
    return [
        {
            "id": row[0],
            "conversation_id": row[1],
            "draft_text": row[2],
            "status": row[3],
            "edited_text": row[4],
            "error_type": row[5],
            "created_at": row[6],
        }
        for row in rows
    ]


def _review_shadow_draft(
    conn: sqlite3.Connection,
    draft_id: int,
    status: str,
    edited_text: str | None = None,
    note: str | None = None,
) -> dict | None:
    """Record a review decision without dispatching the draft.

    Args:
        conn: SQLite connection.
        draft_id: Draft row ID.
        status: One of approved, edited, or rejected.
        edited_text: Replacement text required for edited status.
        note: Optional reviewer note.

    Returns:
        Updated draft summary, or None when the ID is absent.

    Raises:
        ValueError: If the status or edit text is invalid.
    """
    if status not in _REVIEW_STATUSES:
        raise ValueError("status must be approved, edited, or rejected")
    clean_edit = edited_text.strip() if isinstance(edited_text, str) else None
    if status == "edited" and not clean_edit:
        raise ValueError("edited status requires replacement text")
    cursor = conn.execute(
        "UPDATE shadow_drafts SET status = ?, edited_text = ?, review_note = ?, "
        "reviewed_at = ? WHERE id = ? AND status = 'pending'",
        (
            status,
            clean_edit if status == "edited" else None,
            note,
            int(time.time() * 1000),
            draft_id,
        ),
    )
    conn.commit()
    if cursor.rowcount != 1:
        return None
    row = conn.execute(
        "SELECT id, conversation_id, draft_text, status, edited_text, error_type, "
        "created_at FROM shadow_drafts WHERE id = ?",
        (draft_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "conversation_id": row[1],
        "draft_text": row[2],
        "status": row[3],
        "edited_text": row[4],
        "error_type": row[5],
        "created_at": row[6],
    }


def _get_persona_profile(
    conn: sqlite3.Connection,
    platform_id: str | None = None,
) -> dict | None:
    """Return one source persona summary.

    Args:
        conn: SQLite connection.
        platform_id: Optional source platform filter.

    Returns:
        Persona dictionary, or None when no examples exist.
    """
    params: tuple[str, ...] = (platform_id,) if platform_id else ()
    platform_filter = "AND h.platform_id = ? " if platform_id else ""
    rows = conn.execute(
        "SELECT h.platform_id, h.self_id, h.response_text, h.created_at "
        "FROM human_examples h "
        "JOIN conversation_privacy p ON p.conversation_id = h.conversation_id "
        "WHERE p.observe_enabled = 1 AND p.persona_learning_enabled = 1 "
        "AND p.privacy_level != 'excluded' "
        + platform_filter
        + "ORDER BY h.created_at DESC",
        params,
    ).fetchall()
    if not rows:
        return None
    selected_platform, selected_self = rows[0][0], rows[0][1]
    selected = [
        row for row in rows if row[0] == selected_platform and row[1] == selected_self
    ]
    texts = [row[2] for row in selected]
    count = len(texts)
    average_length = round(sum(len(text) for text in texts) / count, 1)
    short_ratio = sum(len(text) <= 12 for text in texts) / count
    question_ratio = sum("?" in text or "？" in text for text in texts) / count
    exclamation_ratio = sum("!" in text or "！" in text for text in texts) / count
    return {
        "platform_id": selected_platform,
        "self_id": selected_self,
        "sample_count": count,
        "summary_text": (
            f"基于 {count} 条已验证回复：平均 {average_length} 字；"
            f"短回复占 {short_ratio:.0%}；问句占 {question_ratio:.0%}；"
            f"感叹句占 {exclamation_ratio:.0%}。"
        ),
        "updated_at": max(row[3] for row in selected),
    }
