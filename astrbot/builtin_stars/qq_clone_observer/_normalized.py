"""Normalized event layer for QQClone passive observer.

Provides deterministic normalization of raw OneBot events into a stable
normalized fact record. The normalized record preserves a direct reference
to the raw row and is stored in a plugin-owned SQLite table. This module is
observation-only and never sends messages or calls external services.
"""

import json
import sqlite3
from typing import Any
from urllib.parse import quote

from astrbot.core.message.components import (
    BaseMessageComponent,
    ComponentType,
    Reply,
)
from astrbot.core.platform.astrbot_message import (
    AstrBotMessage,
    MessageDirection,
    MessageMember,
    MessageType,
)

_NORMALIZED_PARSER_VERSION = "0.1.0"

# Known OneBot segment types mapped to stable normalized kinds
_KNOWN_SEGMENT_KINDS = {
    ComponentType.Plain: "text",
    ComponentType.At: "at",
    ComponentType.Reply: "reply",
    ComponentType.Image: "image",
    ComponentType.Record: "record",
    ComponentType.Video: "video",
    ComponentType.File: "file",
    ComponentType.Face: "face",
    ComponentType.Poke: "poke",
    ComponentType.Share: "share",
    ComponentType.Music: "music",
    ComponentType.Location: "location",
    ComponentType.Contact: "contact",
    ComponentType.Forward: "forward",
    ComponentType.Node: "node",
    ComponentType.Nodes: "nodes",
    ComponentType.Json: "json",
    ComponentType.Unknown: "unknown",
}

_AGENT_PROVENANCE = {"agent", "generated"}
_NON_HUMAN_PROVENANCE = _AGENT_PROVENANCE | {"imported", "forward", "forwarded"}
_KNOWN_RAW_SEGMENT_KINDS = {
    "text",
    "at",
    "reply",
    "image",
    "record",
    "audio",
    "video",
    "file",
    "json",
    "xml",
    "forward",
    "node",
}


def _build_normalized_schema(cursor: sqlite3.Cursor) -> None:
    """Create the normalized_events table with immutable triggers.

    Each raw event produces at most one normalized row, enforced by a UNIQUE
    constraint on raw_event_id. The table is immutable (UPDATE and DELETE are
    rejected by triggers).

    Args:
        cursor: SQLite cursor for the plugin database.
    """
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS normalized_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_event_id    INTEGER NOT NULL UNIQUE,
            parser_version  TEXT NOT NULL,
            platform_id     TEXT NOT NULL,
            self_id         TEXT NOT NULL,
            account_role    TEXT NOT NULL,
            source_event_type TEXT,
            source_event_id TEXT,
            source_message_id TEXT,
            conversation_type TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            sender_id       TEXT,
            sender_display_name TEXT,
            group_id        TEXT,
            direction       TEXT NOT NULL,
            provenance      TEXT,
            actor_provenance TEXT NOT NULL,
            normalized_segments_json TEXT NOT NULL DEFAULT '[]',
            reply_target_message_id TEXT,
            source_occurred_time INTEGER,
            normalized_processing_time TEXT NOT NULL,
            normalization_status TEXT NOT NULL DEFAULT 'normalized',
            normalization_error TEXT,
            FOREIGN KEY (raw_event_id) REFERENCES raw_events(id)
        );

        CREATE INDEX IF NOT EXISTS idx_normalized_conv_id
            ON normalized_events(conversation_id);

        CREATE INDEX IF NOT EXISTS idx_normalized_raw_event
            ON normalized_events(raw_event_id);

        CREATE TRIGGER IF NOT EXISTS trg_normalized_events_no_update
            BEFORE UPDATE ON normalized_events
        BEGIN
            SELECT RAISE(ABORT, 'UPDATE on normalized_events is not allowed');
        END;

        CREATE TRIGGER IF NOT EXISTS trg_normalized_events_no_delete
            BEFORE DELETE ON normalized_events
        BEGIN
            SELECT RAISE(ABORT, 'DELETE on normalized_events is not allowed');
        END;
    """)


def _derive_conversation_type(
    message_obj: AstrBotMessage,
    raw_message: Any,
) -> str:
    """Derive the conversation type from message object and raw payload.

    Priority order:
    1. raw_message.message_type (OneBot ``message_type`` field).
    2. message_obj.type (AstrBot MessageType enum).
    3. presence of message_obj.group as a group indicator.
    4. Fallback to ``unknown``.

    Args:
        message_obj: The AstrBot message object.
        raw_message: The raw message payload (dict or dict-like object).

    Returns:
        ``private``, ``group``, or ``unknown``.
    """
    if raw_message is not None and hasattr(raw_message, "get"):
        try:
            msg_type = raw_message.get("message_type")
            if msg_type == "private":
                return "private"
            if msg_type == "group":
                return "group"
        except (AttributeError, TypeError):
            pass

    if hasattr(message_obj, "type"):
        if message_obj.type == MessageType.GROUP_MESSAGE:
            return "group"
        if message_obj.type == MessageType.FRIEND_MESSAGE:
            return "private"

    if hasattr(message_obj, "group") and message_obj.group is not None:
        return "group"

    return "unknown"


def _build_conversation_id(
    platform_id: str,
    self_id: str,
    conversation_type: str,
    message_obj: AstrBotMessage,
    raw_message: Any,
) -> str:
    """Build a stable non-colliding conversation ID.

    Format per conversation type:

    - **group**: ``{platform_id}:{self_id}:g:{group_id}``
    - **private** (incoming): ``{platform_id}:{self_id}:p:{sender_id}``
    - **private** (outgoing): ``{platform_id}:{self_id}:p:{target_user_id}``
    - **unknown**: ``{platform_id}:{self_id}:u:unknown``

    The scoping by platform_id and self_id ensures that different accounts
    never produce colliding IDs. Private and group IDs are separated by
    the ``p:`` vs ``g:`` prefix to prevent collision.

    Args:
        platform_id: Platform instance ID.
        self_id: Bot self ID.
        conversation_type: Derived conversation type.
        message_obj: The AstrBot message object.
        raw_message: The raw message payload.

    Returns:
        A stable conversation ID string.
    """
    encoded_platform_id = quote(str(platform_id), safe="")
    encoded_self_id = quote(str(self_id), safe="")

    if conversation_type == "group":
        group_id = None
        if hasattr(message_obj, "group") and message_obj.group is not None:
            group_id = message_obj.group.group_id
        if not group_id and raw_message is not None and hasattr(raw_message, "get"):
            try:
                raw_group_id = raw_message.get("group_id")
                group_id = str(raw_group_id) if raw_group_id is not None else None
            except (AttributeError, TypeError):
                pass
        if group_id:
            return (
                f"{encoded_platform_id}:{encoded_self_id}:g:"
                f"{quote(str(group_id), safe='')}"
            )
        return f"{encoded_platform_id}:{encoded_self_id}:u:unknown"

    if conversation_type == "private":
        peer_id = None
        direction = getattr(message_obj, "direction", MessageDirection.UNKNOWN)
        if direction in (MessageDirection.OUTGOING, "outgoing"):
            # Outgoing: use target/user ID from raw_message
            if raw_message is not None and hasattr(raw_message, "get"):
                try:
                    raw_peer_id = raw_message.get("user_id")
                    if raw_peer_id is None:
                        raw_peer_id = raw_message.get("target_id")
                    peer_id = str(raw_peer_id) if raw_peer_id is not None else None
                except (AttributeError, TypeError):
                    pass
        elif direction in (MessageDirection.INCOMING, "incoming"):
            # Incoming: use sender ID
            if hasattr(message_obj, "sender") and message_obj.sender is not None:
                peer_id = message_obj.sender.user_id
        if peer_id:
            return (
                f"{encoded_platform_id}:{encoded_self_id}:p:"
                f"{quote(str(peer_id), safe='')}"
            )
        return f"{encoded_platform_id}:{encoded_self_id}:u:unknown"

    return f"{encoded_platform_id}:{encoded_self_id}:u:unknown"


def _derive_actor_provenance(
    direction: str,
    provenance: str | None,
    account_role: str,
) -> str:
    """Derive actor provenance conservatively.

    Rules applied in order:

    1. Non-outgoing messages (incoming/unknown) become ``unknown``.
    2. **source + outgoing** + provenance not
       ``agent/generated/imported/forwarded`` become ``human``.
    3. **source + outgoing** + provenance explicitly
       ``agent/generated/imported/forwarded`` become ``unknown``.
    4. **clone + outgoing** + explicit ``agent/generated`` becomes ``agent``.
    5. Other clone or fallback cases become ``unknown``.

    Args:
        direction: Message direction string.
        provenance: Provenance string.
        account_role: Configured account role (``source`` or ``clone``).

    Returns:
        ``human``, ``agent``, or ``unknown``.
    """
    if direction != "outgoing":
        return "unknown"

    prov = (provenance or "").strip().lower()

    if account_role == "source":
        if prov not in _NON_HUMAN_PROVENANCE:
            return "human"
        return "unknown"

    if account_role == "clone" and prov in _AGENT_PROVENANCE:
        return "agent"

    return "unknown"


def _normalize_segment(comp: BaseMessageComponent) -> dict:
    """Convert a single AstrBot message component to a normalized segment dict.

    Known segment types are mapped to stable ``kind`` strings. Unknown types
    and unknown fields within known types are preserved safely rather than
    discarded or crashing.

    Args:
        comp: AstrBot message component.

    Returns:
        Normalized segment dict with ``kind`` and ``data`` keys.
    """
    kind = _KNOWN_SEGMENT_KINDS.get(comp.type, "unknown")

    try:
        if comp.type == ComponentType.Plain:
            return {"kind": "text", "data": {"text": comp.text}}
        if comp.type == ComponentType.At:
            qq = str(comp.qq) if hasattr(comp, "qq") else ""
            name = comp.name if hasattr(comp, "name") and comp.name else ""
            return {"kind": "at", "data": {"qq": qq, "name": name}}
        if comp.type == ComponentType.Reply:
            return {"kind": "reply", "data": {"id": str(comp.id)}}
        if comp.type == ComponentType.Image:
            return {
                "kind": "image",
                "data": {
                    "file": comp.file or "",
                    "url": getattr(comp, "url", "") or "",
                    "type": getattr(comp, "_type", "") or "",
                },
            }
        if comp.type == ComponentType.Record:
            return {
                "kind": "record",
                "data": {
                    "file": comp.file or "",
                    "url": getattr(comp, "url", "") or "",
                },
            }
        if comp.type == ComponentType.Video:
            return {
                "kind": "video",
                "data": {
                    "file": comp.file or "",
                    "url": getattr(comp, "url", "") or "",
                },
            }
        if comp.type == ComponentType.File:
            return {
                "kind": "file",
                "data": {
                    "name": comp.name or "",
                    "file": getattr(comp, "file_", "") or "",
                    "url": getattr(comp, "url", "") or "",
                },
            }
        if comp.type == ComponentType.Face:
            return {"kind": "face", "data": {"id": comp.id}}
        if comp.type == ComponentType.Poke:
            poke_type = str(comp._type) if hasattr(comp, "_type") else ""
            target_id = ""
            if hasattr(comp, "target_id"):
                try:
                    target_id = comp.target_id() or ""
                except Exception:
                    pass
            return {
                "kind": "poke",
                "data": {"type": poke_type, "id": target_id},
            }
        if comp.type == ComponentType.Share:
            return {
                "kind": "share",
                "data": {
                    "url": comp.url,
                    "title": comp.title,
                    "content": getattr(comp, "content", "") or "",
                },
            }
        if comp.type == ComponentType.Music:
            return {
                "kind": "music",
                "data": {
                    "type": comp._type if hasattr(comp, "_type") else "",
                    "id": comp.id,
                    "url": getattr(comp, "url", "") or "",
                },
            }
        if comp.type == ComponentType.Location:
            return {
                "kind": "location",
                "data": {
                    "lat": comp.lat,
                    "lon": comp.lon,
                    "title": getattr(comp, "title", "") or "",
                },
            }
        if comp.type == ComponentType.Contact:
            return {
                "kind": "contact",
                "data": {
                    "type": getattr(comp, "_type", "") or "",
                    "id": getattr(comp, "id", None),
                },
            }
        if comp.type == ComponentType.Forward:
            return {"kind": "forward", "data": {"id": comp.id}}
        if comp.type == ComponentType.Node:
            return {
                "kind": "node",
                "data": {
                    "user_id": str(getattr(comp, "uin", "0") or "0"),
                    "nickname": getattr(comp, "name", "") or "",
                    "content": [_normalize_segment(sub) for sub in comp.content],
                },
            }
        if comp.type == ComponentType.Nodes:
            return {
                "kind": "nodes",
                "data": {
                    "nodes": [_normalize_segment(n) for n in comp.nodes],
                },
            }
        if comp.type == ComponentType.Json:
            return {"kind": "json", "data": {"data": comp.data}}
    except Exception:
        # If any known segment extraction fails, fall through to unknown
        pass

    # Preserve unknown segment safely
    try:
        raw = comp.toDict() if hasattr(comp, "toDict") else {"type": kind, "data": {}}
    except Exception:
        raw = {"type": kind, "data": {}}
    return {"kind": "unknown", "data": raw}


def _normalize_segments(message_obj: AstrBotMessage) -> list[dict]:
    """Convert AstrBot message segments to an ordered list of normalized dicts.

    The order of segments is preserved from the original message chain.
    Returns an empty list if the message has no segments.

    Args:
        message_obj: The AstrBot message object.

    Returns:
        List of normalized segment dicts.
    """
    segments = getattr(message_obj, "message", None)
    if not segments or not isinstance(segments, list):
        return []
    return [_normalize_segment(comp) for comp in segments]


def _normalize_raw_segments(raw_message: Any) -> list[dict]:
    """Normalize raw OneBot segments without discarding unknown fields.

    Args:
        raw_message: Raw OneBot event payload.

    Returns:
        Ordered normalized raw segments, or an empty list when unavailable.
    """
    if raw_message is None or not hasattr(raw_message, "get"):
        return []
    try:
        raw_segments = raw_message.get("message")
    except (AttributeError, TypeError):
        return []
    if not isinstance(raw_segments, list):
        return []

    normalized = []
    for segment in raw_segments:
        if not isinstance(segment, dict):
            normalized.append({"kind": "unknown", "data": {"value": segment}})
            continue
        raw_kind = segment.get("type")
        kind = str(raw_kind).lower() if raw_kind else "unknown"
        data = segment.get("data", {})
        if not isinstance(data, dict):
            data = {"value": data}
        safe_data = json.loads(json.dumps(data, ensure_ascii=False, default=str))
        if kind in _KNOWN_RAW_SEGMENT_KINDS:
            extra_fields = {
                key: value
                for key, value in segment.items()
                if key not in {"type", "data"}
            }
            if extra_fields:
                safe_data["_extra_fields"] = json.loads(
                    json.dumps(extra_fields, ensure_ascii=False, default=str)
                )
            normalized.append({"kind": kind, "data": safe_data})
            continue

        safe_segment = json.loads(json.dumps(segment, ensure_ascii=False, default=str))
        normalized.append(
            {
                "kind": "unknown",
                "data": {
                    "original_type": raw_kind,
                    "original_segment": safe_segment,
                },
            }
        )
    return normalized


def _extract_reply_target(
    message_obj: AstrBotMessage,
    raw_message: Any,
) -> str | None:
    """Extract the reply target message ID if present.

    Checks AstrBot message components for a Reply segment, then falls back
    to the raw_message ``reply`` field (OneBot format).

    Args:
        message_obj: The AstrBot message object.
        raw_message: The raw message payload.

    Returns:
        Reply target message ID string, or None.
    """
    segments = getattr(message_obj, "message", None)
    if segments and isinstance(segments, list):
        for comp in segments:
            if isinstance(comp, Reply):
                return str(comp.id)

    if raw_message is not None and hasattr(raw_message, "get"):
        try:
            reply_id = raw_message.get("reply")
            if reply_id is not None:
                return str(reply_id)

            raw_segments = raw_message.get("message")
            if isinstance(raw_segments, list):
                for segment in raw_segments:
                    if not isinstance(segment, dict) or segment.get("type") != "reply":
                        continue
                    data = segment.get("data")
                    if isinstance(data, dict) and data.get("id") is not None:
                        return str(data["id"])
        except (AttributeError, TypeError):
            pass

    return None


def _extract_source_occurred_time(
    message_obj: AstrBotMessage,
    raw_message: Any,
) -> int | None:
    """Extract the source occurred timestamp.

    Prefers the raw_message ``time`` field (OneBot UTC timestamp), then
    falls back to message_obj.timestamp.

    Args:
        message_obj: The AstrBot message object.
        raw_message: The raw message payload.

    Returns:
        Unix timestamp integer, or None.
    """
    if raw_message is not None and hasattr(raw_message, "get"):
        try:
            ts = raw_message.get("time")
            if ts is not None:
                return int(ts)
        except (AttributeError, TypeError, ValueError, OverflowError):
            pass

    ts = getattr(message_obj, "timestamp", None)
    if ts is not None:
        try:
            return int(ts)
        except (TypeError, ValueError, OverflowError):
            pass

    return None


def _normalize_event(
    raw_event_id: int,
    platform_id: str,
    self_id: str,
    account_role: str,
    source_event_type: str | None,
    source_event_id: str | None,
    source_message_id: str | None,
    direction: str,
    provenance: str | None,
    message_obj: AstrBotMessage,
    raw_message: Any,
    normalized_processing_time: str | None = None,
) -> dict:
    """Normalize a raw event into a normalized fact record dict.

    The normalization is deterministic: given the same raw event parameters
    and account metadata, it produces the same normalized output.

    Args:
        raw_event_id: The raw_events table row ID.
        platform_id: Platform instance ID.
        self_id: Bot self ID.
        account_role: Configured account role.
        source_event_type: Event type from the adapter.
        source_event_id: Stable source event ID.
        source_message_id: Message ID from the source.
        direction: Message direction string.
        provenance: Provenance string.
        message_obj: The AstrBot message object.
        raw_message: The raw message payload.

    Returns:
        Dict suitable for insertion into normalized_events table.
    """
    conversation_type = _derive_conversation_type(message_obj, raw_message)

    conversation_id = _build_conversation_id(
        platform_id,
        self_id,
        conversation_type,
        message_obj,
        raw_message,
    )
    if conversation_id.endswith(":u:unknown"):
        conversation_type = "unknown"

    sender_id = None
    sender_display_name = None
    if raw_message is not None and hasattr(raw_message, "get"):
        try:
            raw_sender = raw_message.get("sender")
            if isinstance(raw_sender, dict):
                raw_sender_id = raw_sender.get("user_id")
                sender_id = str(raw_sender_id) if raw_sender_id is not None else None
                sender_display_name = (
                    raw_sender.get("card") or raw_sender.get("nickname") or None
                )
        except (AttributeError, TypeError):
            pass
    if (
        sender_id is None
        and hasattr(message_obj, "sender")
        and message_obj.sender is not None
    ):
        sender_id = message_obj.sender.user_id
        sender_display_name = message_obj.sender.nickname

    group_id = None
    if conversation_type == "group":
        if hasattr(message_obj, "group") and message_obj.group is not None:
            group_id = message_obj.group.group_id
        if not group_id and raw_message is not None and hasattr(raw_message, "get"):
            try:
                raw_group_id = raw_message.get("group_id")
                group_id = str(raw_group_id) if raw_group_id is not None else None
            except (AttributeError, TypeError):
                pass

    actor_provenance = _derive_actor_provenance(
        direction,
        provenance,
        account_role,
    )

    normalized_segments = _normalize_raw_segments(raw_message)
    if not normalized_segments:
        normalized_segments = _normalize_segments(message_obj)
    reply_target = _extract_reply_target(message_obj, raw_message)
    source_occurred_time = _extract_source_occurred_time(message_obj, raw_message)

    return {
        "raw_event_id": raw_event_id,
        "parser_version": _NORMALIZED_PARSER_VERSION,
        "platform_id": platform_id,
        "self_id": self_id,
        "account_role": account_role,
        "source_event_type": source_event_type,
        "source_event_id": source_event_id,
        "source_message_id": source_message_id,
        "conversation_type": conversation_type,
        "conversation_id": conversation_id,
        "sender_id": sender_id,
        "sender_display_name": sender_display_name,
        "group_id": group_id,
        "direction": direction,
        "provenance": provenance,
        "actor_provenance": actor_provenance,
        "normalized_segments_json": json.dumps(
            normalized_segments, ensure_ascii=False, sort_keys=True
        ),
        "reply_target_message_id": reply_target,
        "source_occurred_time": source_occurred_time,
        "normalized_processing_time": normalized_processing_time
        if normalized_processing_time is not None
        else str(source_occurred_time or 0),
        "normalization_status": "normalized",
        "normalization_error": None,
    }


def _insert_normalized_event(
    conn: sqlite3.Connection,
    normalized: dict,
) -> bool:
    """Insert a normalized event row, ignoring duplicates.

    Args:
        conn: SQLite connection.
        normalized: Normalized event dict.

    Returns:
        True if a new row was inserted, False if duplicate was ignored.
    """
    cursor = conn.execute(
        "INSERT INTO normalized_events "
        "(raw_event_id, parser_version, platform_id, self_id, account_role, "
        "source_event_type, source_event_id, source_message_id, "
        "conversation_type, conversation_id, sender_id, sender_display_name, "
        "group_id, direction, provenance, actor_provenance, "
        "normalized_segments_json, reply_target_message_id, "
        "source_occurred_time, normalized_processing_time, "
        "normalization_status, normalization_error) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(raw_event_id) DO NOTHING",
        (
            normalized["raw_event_id"],
            normalized["parser_version"],
            normalized["platform_id"],
            normalized["self_id"],
            normalized["account_role"],
            normalized["source_event_type"],
            normalized["source_event_id"],
            normalized["source_message_id"],
            normalized["conversation_type"],
            normalized["conversation_id"],
            normalized["sender_id"],
            normalized["sender_display_name"],
            normalized["group_id"],
            normalized["direction"],
            normalized["provenance"],
            normalized["actor_provenance"],
            normalized["normalized_segments_json"],
            normalized["reply_target_message_id"],
            normalized["source_occurred_time"],
            normalized["normalized_processing_time"],
            normalized["normalization_status"],
            normalized["normalization_error"],
        ),
    )
    conn.commit()
    return cursor.rowcount == 1


def _normalize_stored_raw_event(
    conn: sqlite3.Connection,
    raw_event_id: int,
) -> bool:
    """Normalize one immutable raw row and store its derived fact.

    Args:
        conn: SQLite connection.
        raw_event_id: Existing raw event row ID.

    Returns:
        True when a normalized row was inserted, or False when it already exists.

    Raises:
        ValueError: If the raw row is missing or its payload is not a JSON object.
    """
    cursor = conn.execute(
        "SELECT platform_id, self_id, account_role, source_event_type, "
        "source_event_id, source_message_id, message_direction, provenance, "
        "payload_json, received_time FROM raw_events WHERE id = ?",
        (raw_event_id,),
    )
    row = cursor.fetchone()
    if row is None:
        raise ValueError("raw event not found")

    raw_message = json.loads(row[8])
    if not isinstance(raw_message, dict):
        raw_message = {}

    message_obj = AstrBotMessage()
    message_obj.direction = (
        MessageDirection.OUTGOING
        if row[6] == "outgoing"
        else MessageDirection.INCOMING
        if row[6] == "incoming"
        else MessageDirection.UNKNOWN
    )
    direction = message_obj.direction.value
    raw_sender = raw_message.get("sender")
    if isinstance(raw_sender, dict) and raw_sender.get("user_id") is not None:
        message_obj.sender = MessageMember(
            user_id=str(raw_sender["user_id"]),
            nickname=raw_sender.get("card") or raw_sender.get("nickname"),
        )
    message_obj.message = []

    normalized = _normalize_event(
        raw_event_id,
        row[0],
        row[1],
        row[2],
        row[3],
        row[4],
        row[5],
        direction,
        row[7],
        message_obj,
        raw_message,
        normalized_processing_time=row[9],
    )
    return _insert_normalized_event(conn, normalized)
