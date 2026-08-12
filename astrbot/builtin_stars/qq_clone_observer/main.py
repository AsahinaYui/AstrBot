"""QQClone passive observer plugin.

Observes raw message events from configured source/clone QQ accounts and
persists them to a plugin-owned SQLite database. This plugin is observation-only
and never sends messages or calls external services.
"""

import asyncio
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.core import logger
from astrbot.core.platform.astrbot_message import MessageDirection
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

from ._normalized import (
    _build_normalized_schema,
    _insert_normalized_event,
    _normalize_event,
)

_PARSER_VERSION = "0.1.0"
_DB_FILENAME = "raw_events.db"


def _safe_json_serialize(obj: Any, max_depth: int = 10) -> str:
    """Safely serialize an arbitrary object to JSON.

    Handles dicts, lists, primitives, and falls back to ``str(obj)`` for
    unsupported types (e.g. aiocqhttp Event objects).

    Args:
        obj: Object to serialize.
        max_depth: Maximum recursion depth to prevent infinite loops.

    Returns:
        JSON string representation of the object.
    """
    if max_depth <= 0:
        return json.dumps(str(obj), ensure_ascii=False)

    if isinstance(obj, (str, int, float, bool, type(None))):
        return json.dumps(obj, ensure_ascii=False)

    if isinstance(obj, bytes):
        return json.dumps(obj.hex(), ensure_ascii=False)

    if isinstance(obj, dict):
        items = {}
        for k, v in obj.items():
            k_str = str(k)
            items[k_str] = json.loads(_safe_json_serialize(v, max_depth - 1))
        return json.dumps(items, ensure_ascii=False, sort_keys=True)

    if isinstance(obj, (list, tuple)):
        items = [json.loads(_safe_json_serialize(item, max_depth - 1)) for item in obj]
        return json.dumps(items, ensure_ascii=False)

    # Fallback for unsupported types (e.g. aiocqhttp Event, custom objects)
    return json.dumps(str(obj), ensure_ascii=False)


def _compute_payload_sha256(payload_json: str) -> str:
    """Compute SHA-256 of a JSON payload string.

    Args:
        payload_json: JSON string to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _canonical_fingerprint(
    platform_id: str,
    self_id: str,
    direction: str | None,
    provenance: str | None,
    payload_sha256: str,
) -> str:
    """Compute a SHA-256 fingerprint over stable canonical identity fields.

    Used as a fallback idempotency key when no stable source event ID
    or message ID is available. Does not include message text or timestamp.

    Args:
        platform_id: Platform instance ID.
        self_id: Bot self ID.
        direction: Message direction string.
        provenance: Provenance string.
        payload_sha256: SHA-256 of the payload JSON.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    raw = (
        f"{platform_id}|{self_id}|{direction or ''}|{provenance or ''}|{payload_sha256}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _build_schema(cursor: sqlite3.Cursor) -> None:
    """Create the raw_events table, UNIQUE indexes, and immutable triggers.

    Args:
        cursor: SQLite cursor for the plugin database.
    """
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS raw_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            platform_id     TEXT NOT NULL,
            self_id         TEXT NOT NULL,
            account_role    TEXT NOT NULL,
            source_event_type TEXT,
            source_event_id TEXT,
            source_message_id TEXT,
            message_direction TEXT NOT NULL,
            provenance      TEXT,
            payload_json    TEXT NOT NULL,
            payload_sha256  TEXT NOT NULL,
            stable_fingerprint TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            parser_version  TEXT NOT NULL,
            received_time   TEXT NOT NULL,
            processing_status TEXT NOT NULL DEFAULT 'accepted'
        );

        CREATE INDEX IF NOT EXISTS idx_raw_events_event_id
            ON raw_events(platform_id, self_id, source_event_id)
            WHERE source_event_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_raw_events_platform_self_msg
            ON raw_events(platform_id, self_id, source_message_id)
            WHERE source_message_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_raw_events_fingerprint
            ON raw_events(stable_fingerprint);

        CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_events_idempotency_key
            ON raw_events(idempotency_key);

        CREATE TRIGGER IF NOT EXISTS trg_raw_events_no_update
            BEFORE UPDATE ON raw_events
        BEGIN
            SELECT RAISE(ABORT, 'UPDATE on raw_events is not allowed');
        END;

        CREATE TRIGGER IF NOT EXISTS trg_raw_events_no_delete
            BEFORE DELETE ON raw_events
        BEGIN
            SELECT RAISE(ABORT, 'DELETE on raw_events is not allowed');
        END;
    """)

    _build_normalized_schema(cursor)


def _insert_raw_event(
    conn: sqlite3.Connection,
    platform_id: str,
    self_id: str,
    account_role: str,
    source_event_type: str | None,
    source_event_id: str | None,
    source_message_id: str | None,
    message_direction: str,
    provenance: str | None,
    payload_json: str,
    payload_sha256: str,
    stable_fingerprint: str,
    idempotency_key: str,
) -> int | None:
    """Insert a raw event row, ignoring duplicates.

    Uses INSERT OR IGNORE so that duplicate rows (enforced by UNIQUE
    indexes) are silently skipped.

    Args:
        conn: SQLite connection.
        platform_id: Platform instance ID.
        self_id: Bot self ID.
        account_role: Configured account role.
        source_event_type: Event type from the adapter.
        source_event_id: Stable source event ID (e.g. OneBot event_id).
        source_message_id: Message ID from the source.
        message_direction: Direction string.
        provenance: Provenance string.
        payload_json: Serialized JSON payload.
        payload_sha256: SHA-256 of the payload.
        stable_fingerprint: SHA-256 over canonical identity fields.
        idempotency_key: Key selected from the highest available identity tier.

    Returns:
        The row ID if a new row was inserted, None if duplicate was ignored.
    """
    cursor = conn.execute(
        "INSERT OR IGNORE INTO raw_events "
        "(platform_id, self_id, account_role, source_event_type, "
        "source_event_id, source_message_id, message_direction, provenance, "
        "payload_json, payload_sha256, stable_fingerprint, idempotency_key, "
        "parser_version, received_time, processing_status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            platform_id,
            self_id,
            account_role,
            source_event_type,
            source_event_id,
            source_message_id,
            message_direction,
            provenance,
            payload_json,
            payload_sha256,
            stable_fingerprint,
            idempotency_key,
            _PARSER_VERSION,
            str(int(time.time())),
            "accepted",
        ),
    )
    conn.commit()
    return cursor.lastrowid if cursor.rowcount == 1 else None


def _check_existing_by_event_id(
    conn: sqlite3.Connection,
    platform_id: str,
    self_id: str,
    source_event_id: str,
) -> bool:
    """Check if a raw event with the given source event ID exists for this account.

    Args:
        conn: SQLite connection.
        platform_id: Platform instance ID.
        self_id: Bot self ID.
        source_event_id: Stable source event ID.

    Returns:
        True if an existing row is found.
    """
    cursor = conn.execute(
        "SELECT 1 FROM raw_events WHERE platform_id = ? AND self_id = ? "
        "AND source_event_id = ? LIMIT 1",
        (platform_id, self_id, source_event_id),
    )
    return cursor.fetchone() is not None


def _check_existing_by_platform_and_msg_id(
    conn: sqlite3.Connection,
    platform_id: str,
    self_id: str,
    source_message_id: str,
) -> bool:
    """Check if a raw event exists for this account + message ID.

    Args:
        conn: SQLite connection.
        platform_id: Platform instance ID.
        self_id: Bot self ID.
        source_message_id: Source message ID.

    Returns:
        True if an existing row is found.
    """
    cursor = conn.execute(
        "SELECT 1 FROM raw_events WHERE platform_id = ? AND self_id = ? "
        "AND source_message_id = ? LIMIT 1",
        (platform_id, self_id, source_message_id),
    )
    return cursor.fetchone() is not None


def _check_existing_by_fingerprint(
    conn: sqlite3.Connection,
    fingerprint: str,
) -> bool:
    """Check if a raw event with the given stable fingerprint exists.

    Args:
        conn: SQLite connection.
        fingerprint: Stable fingerprint to check.

    Returns:
        True if an existing row is found.
    """
    cursor = conn.execute(
        "SELECT 1 FROM raw_events WHERE stable_fingerprint = ? LIMIT 1",
        (fingerprint,),
    )
    return cursor.fetchone() is not None


class QQCloneObserver(Star):
    """Observes raw message events from configured source/clone QQ accounts.

    This plugin is observation-only. It persists raw events to a plugin-owned
    SQLite database and tracks in-memory health state for each configured
    account. It never sends messages or calls external services.
    """

    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self._config = config if config is not None else AstrBotConfig()
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

        # In-memory health state per platform_id
        self._health: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        """Initialize the SQLite database and schema."""
        data_dir = Path(get_astrbot_plugin_data_path()) / "qq_clone_observer"
        await asyncio.to_thread(data_dir.mkdir, parents=True, exist_ok=True)
        db_path = data_dir / _DB_FILENAME

        self._conn = await asyncio.to_thread(
            sqlite3.connect,
            str(db_path),
            check_same_thread=False,
        )
        await asyncio.to_thread(self._conn.execute, "PRAGMA journal_mode=WAL")
        await asyncio.to_thread(self._conn.execute, "PRAGMA busy_timeout=5000")
        await asyncio.to_thread(self._conn.execute, "PRAGMA foreign_keys=ON")
        await asyncio.to_thread(_build_schema, self._conn.cursor())

        # Rebuild health state from configuration
        self._rebuild_health()

        logger.info(
            "QQCloneObserver: SQLite database initialized at %s",
            db_path,
        )

    async def terminate(self) -> None:
        """Close the SQLite connection."""
        if self._conn is not None:
            await asyncio.to_thread(self._conn.close)
            self._conn = None

    def _rebuild_health(self) -> None:
        """Rebuild in-memory health state from current configuration."""
        self._health = {}
        accounts = self._config.get("accounts", [])
        if not isinstance(accounts, list):
            accounts = []
        for entry in accounts:
            if not isinstance(entry, dict):
                continue
            pid = entry.get("platform_id", "")
            if not pid:
                continue
            self._health[pid] = {
                "configured_role": entry.get("role", "source"),
                "expected_self_id": entry.get("self_id", ""),
                "observed_self_id": None,
                "accepted_count": 0,
                "duplicate_count": 0,
                "rejected_count": 0,
                "last_event_time": None,
                "last_rejection_or_error": None,
            }

    def get_health(self) -> dict[str, dict[str, Any]]:
        """Return a read-only snapshot of the current health state.

        Returns:
            A dict keyed by platform_id with health counters and status.
        """
        # Return a shallow copy to prevent mutation
        return {pid: dict(state) for pid, state in self._health.items()}

    def _resolve_account_role(
        self,
        platform_id: str,
        self_id: str,
    ) -> tuple[str | None, str | None, str | None]:
        """Resolve the account role from configuration.

        Performs fail-closed identity checks in order:
        1. Platform ID must be configured exactly once (no duplicates).
        2. Account must be enabled.
        3. Self ID must match the configured expected self ID.
        4. Role must be valid (source or clone).

        Args:
            platform_id: Platform instance ID from the event.
            self_id: Self ID from the event.

        Returns:
            A tuple of (role, expected_self_id, rejection_reason). If the
            account is accepted, role is the configured role and
            rejection_reason is None. If rejected, role is None and
            rejection_reason indicates the reason.
        """
        accounts = self._config.get("accounts", [])
        if not isinstance(accounts, list):
            accounts = []

        # Check for duplicate platform_ids in configuration
        pids = [
            entry.get("platform_id", "")
            for entry in accounts
            if isinstance(entry, dict) and entry.get("platform_id")
        ]
        if len(pids) != len(set(pids)):
            if sum(1 for pid in pids if pid == platform_id) > 1:
                return (
                    None,
                    None,
                    f"duplicate platform_id in configuration: {platform_id}",
                )

        for entry in accounts:
            if not isinstance(entry, dict):
                continue
            pid = entry.get("platform_id", "")
            if pid != platform_id:
                continue

            # Check enabled
            if not entry.get("enabled", True):
                return None, None, "account disabled"

            # Check self_id match
            expected_self_id = entry.get("self_id", "")
            if self_id != expected_self_id:
                return (
                    None,
                    expected_self_id,
                    f"self_id mismatch: expected {expected_self_id}, got {self_id}",
                )

            # Resolve role
            role = entry.get("role", "source")
            if role not in ("source", "clone"):
                return None, expected_self_id, f"invalid role: {role}"

            return role, expected_self_id, None

        return None, None, "unknown platform_id"

    @filter.on_message_observed()
    async def on_message_observed(self, event: AstrMessageEvent) -> None:
        """Handle observed messages from all configured accounts.

        This is the main observation handler. It resolves the account role,
        performs identity checks, deduplicates, and persists the raw event.

        Args:
            event: The observed message event.
        """
        message_obj = event.message_obj
        platform_id = event.get_platform_id()
        self_id = event.get_self_id()

        # Resolve account role from configuration (fail-closed)
        role, expected_self_id, rejection_reason = self._resolve_account_role(
            platform_id,
            self_id,
        )

        if role is None:
            # Reject - update health state and log warning
            async with self._lock:
                health = self._health.get(platform_id)
                if health:
                    health["rejected_count"] += 1
                    health["last_rejection_or_error"] = rejection_reason
                    health["observed_self_id"] = self_id
                else:
                    # Unknown platform - still track it
                    self._health[platform_id] = {
                        "configured_role": "unknown",
                        "expected_self_id": "unknown",
                        "observed_self_id": self_id,
                        "accepted_count": 0,
                        "duplicate_count": 0,
                        "rejected_count": 1,
                        "last_event_time": None,
                        "last_rejection_or_error": rejection_reason,
                    }

            if rejection_reason:
                logger.warning(
                    "QQCloneObserver: rejected event from %s/%s: %s",
                    platform_id,
                    self_id,
                    rejection_reason,
                )
            return

        # Extract event metadata
        direction = getattr(message_obj, "direction", MessageDirection.UNKNOWN)
        direction_str = (
            direction.value
            if isinstance(direction, MessageDirection)
            else str(direction)
        )
        provenance = getattr(message_obj, "provenance", "unknown") or "unknown"
        source_message_id = getattr(message_obj, "message_id", None) or None

        # Determine source event type
        raw_message = getattr(message_obj, "raw_message", None)
        source_event_type = None
        if raw_message is not None:
            if isinstance(raw_message, dict):
                source_event_type = raw_message.get("post_type")
            elif hasattr(raw_message, "get"):
                try:
                    source_event_type = raw_message.get("post_type")
                except (AttributeError, TypeError):
                    pass

        # Extract source event ID from raw message
        source_event_id = None
        if raw_message is not None and hasattr(raw_message, "get"):
            try:
                source_event_id = raw_message.get("event_id") or None
            except (AttributeError, TypeError):
                pass

        # Serialize payload
        payload_json = _safe_json_serialize(raw_message)
        payload_sha256 = _compute_payload_sha256(payload_json)

        # Compute stable fingerprint for fallback idempotency
        stable_fingerprint = _canonical_fingerprint(
            platform_id,
            self_id,
            direction_str,
            provenance,
            payload_sha256,
        )
        if source_event_id:
            idempotency_key = f"event:{platform_id}:{self_id}:{source_event_id}"
        elif source_message_id:
            idempotency_key = f"message:{platform_id}:{self_id}:{source_message_id}"
        else:
            idempotency_key = f"fingerprint:{stable_fingerprint}"

        # Check idempotency with the lock
        async with self._lock:
            if self._conn is None:
                logger.error("QQCloneObserver: database not initialized")
                return

            # Idempotency check priority order
            is_duplicate = False

            # 1. Stable source event ID (scoped by platform + self)
            if source_event_id:
                is_duplicate = await self._check_duplicate_by_event_id(
                    platform_id,
                    self_id,
                    source_event_id,
                )

            # 2. Account/platform + self_id + source message ID
            elif source_message_id:
                is_duplicate = await self._check_duplicate_by_platform_and_msg_id(
                    platform_id,
                    self_id,
                    source_message_id,
                )

            # 3. Stable fingerprint fallback
            else:
                is_duplicate = await self._check_duplicate_by_fingerprint(
                    stable_fingerprint,
                )

            if is_duplicate:
                # Update health state
                health = self._health.get(platform_id)
                if health:
                    health["duplicate_count"] += 1
                    health["last_event_time"] = time.time()
                return

            # Persist to SQLite via executor
            try:
                loop = asyncio.get_running_loop()
                raw_event_id = await loop.run_in_executor(
                    None,
                    _insert_raw_event,
                    self._conn,
                    platform_id,
                    self_id,
                    role,
                    source_event_type,
                    source_event_id,
                    source_message_id,
                    direction_str,
                    provenance,
                    payload_json,
                    payload_sha256,
                    stable_fingerprint,
                    idempotency_key,
                )
                if raw_event_id is None:
                    # DB-level duplicate (race condition safety net)
                    health = self._health.get(platform_id)
                    if health:
                        health["duplicate_count"] += 1
                        health["last_event_time"] = time.time()
                    return
            except Exception as e:
                logger.error(
                    "QQCloneObserver: failed to persist event: %s",
                    e,
                )
                # Update health state with error
                health = self._health.get(platform_id)
                if health:
                    health["rejected_count"] += 1
                    health["last_rejection_or_error"] = str(e)
                return

            # Normalize immediately after raw insert succeeds
            try:
                loop = asyncio.get_running_loop()
                normalized = await loop.run_in_executor(
                    None,
                    _normalize_event,
                    raw_event_id,
                    platform_id,
                    self_id,
                    role,
                    source_event_type,
                    source_event_id,
                    source_message_id,
                    direction_str,
                    provenance,
                    message_obj,
                    raw_message,
                )
                await loop.run_in_executor(
                    None,
                    _insert_normalized_event,
                    self._conn,
                    normalized,
                )
            except Exception as e:
                logger.error(
                    "QQCloneObserver: failed to normalize event (raw_id=%s): %s",
                    raw_event_id,
                    e,
                )
                # Record error in health but raw row is preserved
                health = self._health.get(platform_id)
                if health:
                    health["last_rejection_or_error"] = f"normalization_error: {e}"

            # Update health state
            health = self._health.get(platform_id)
            if health:
                health["accepted_count"] += 1
                health["last_event_time"] = time.time()
                health["observed_self_id"] = self_id

    async def _check_duplicate_by_event_id(
        self,
        platform_id: str,
        self_id: str,
        source_event_id: str,
    ) -> bool:
        """Check if a source event ID already exists for this account.

        Args:
            platform_id: Platform instance ID.
            self_id: Bot self ID.
            source_event_id: Stable source event ID.

        Returns:
            True if duplicate found.
        """
        if self._conn is None:
            return False
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            _check_existing_by_event_id,
            self._conn,
            platform_id,
            self_id,
            source_event_id,
        )

    async def _check_duplicate_by_platform_and_msg_id(
        self,
        platform_id: str,
        self_id: str,
        source_message_id: str,
    ) -> bool:
        """Check if a platform + self_id + message ID pair already exists.

        Args:
            platform_id: Platform instance ID.
            self_id: Bot self ID.
            source_message_id: Source message ID.

        Returns:
            True if duplicate found.
        """
        if self._conn is None:
            return False
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            _check_existing_by_platform_and_msg_id,
            self._conn,
            platform_id,
            self_id,
            source_message_id,
        )

    async def _check_duplicate_by_fingerprint(self, fingerprint: str) -> bool:
        """Check if a stable fingerprint already exists.

        Args:
            fingerprint: Stable fingerprint to check.

        Returns:
            True if duplicate found.
        """
        if self._conn is None:
            return False
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            _check_existing_by_fingerprint,
            self._conn,
            fingerprint,
        )
