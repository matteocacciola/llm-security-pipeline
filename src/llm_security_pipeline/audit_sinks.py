"""
audit_sinks.py
Two more places for the audit stream to go.

The pipeline ships with stdout (fine for a demo) and a Redis Stream (fine
when Redis is already there). Two common deployments had nothing: a
service with no Redis that still needs a durable, tailable record, and a
service whose logs already flow through OpenTelemetry and would rather the
audit events flowed with them than through a second pipeline.

Both keep the record shape the other loggers use — `{"ts", "event",
"data"}` with `data` carrying `schema_version` — so a consumer written
against one can read another.

--- Why the file logger rotates per process and not per file ---------------

Rotation is rename-and-reopen, and two processes renaming the same file
race: one loses its handle to a file the other just moved, and a window
of events lands in a file nobody will read. So `{pid}` is available in the
path template and used by default. One file per process is a folder of
files; that is what `tail -F` and log shippers are for, and it is the
only shape that is correct without a lock across processes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

from .pipeline import AuditLogger


class FileAuditLogger(AuditLogger):
    """Append audit events as JSON lines to a file, rotating by size.

        SecurityPipeline(audit_logger=FileAuditLogger("/var/log/guard/audit-{pid}.jsonl"))

    Writes are serialized with a lock and done inline: a line is small and
    the file is local, so an executor hop would cost more than the write.
    `fsync` is off by default — a crash loses the last few lines, which is
    the usual trade for a log — and can be turned on for audit trails that
    must survive a power cut.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        max_bytes: int = 64 * 1024 * 1024,
        backup_count: int = 5,
        fsync: bool = False,
    ):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive.")
        if backup_count < 0:
            raise ValueError("backup_count cannot be negative.")
        self.path = Path(str(path).format(pid=os.getpid()))
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.fsync = fsync
        self._lock = asyncio.Lock()
        self._handle = None
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _open(self):
        if self._handle is None:
            self._handle = open(self.path, "a", encoding="utf-8")
        return self._handle

    def _rotate(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self.backup_count == 0:
            self.path.unlink(missing_ok=True)
            return
        # audit.jsonl.4 -> .5 (dropped), ..., audit.jsonl -> audit.jsonl.1
        for index in range(self.backup_count - 1, 0, -1):
            older = self.path.with_name(f"{self.path.name}.{index}")
            if older.exists():
                older.replace(self.path.with_name(f"{self.path.name}.{index + 1}"))
        if self.path.exists():
            self.path.replace(self.path.with_name(f"{self.path.name}.1"))

    async def log(self, event_type: str, data: dict) -> None:
        line = json.dumps(
            {"ts": time.time(), "event": event_type, "data": data},
            ensure_ascii=False, default=str,
        ) + "\n"
        async with self._lock:
            handle = self._open()
            if handle.tell() + len(line.encode("utf-8")) > self.max_bytes:
                self._rotate()
                handle = self._open()
            handle.write(line)
            handle.flush()
            if self.fsync:
                os.fsync(handle.fileno())

    async def aclose(self) -> None:
        async with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None


class PythonLoggingAuditLogger(AuditLogger):
    """Emit audit events through the standard `logging` module.

    This is how they reach OpenTelemetry: OTel's logging handler bridges
    stdlib logging to the OTel logs pipeline, so attaching it to the logger
    this uses sends every audit event where the application's other logs
    already go, with the same resource and trace correlation. The handler
    lives in `opentelemetry-instrumentation-logging` (current) or, older
    and now deprecated, in `opentelemetry.sdk._logs`; either works, the
    mechanism is stdlib logging.

        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.instrumentation.logging import LoggingHandler  # or opentelemetry.sdk._logs
        provider = LoggerProvider(); ...add your exporter...
        logging.getLogger("llm_security_pipeline.audit").addHandler(
            LoggingHandler(logger_provider=provider)
        )
        SecurityPipeline(audit_logger=PythonLoggingAuditLogger())

    Going through `logging` rather than the OTel API directly keeps the
    library free of a hard OTel dependency and lets a deployment without
    OTel still route audit events through whatever handler it prefers.
    The event type and every field of `data` are attached as record
    attributes (prefixed `audit.`), so a backend can index them without
    parsing the message; the message itself is the JSON line, so a plain
    file handler produces the same shape as FileAuditLogger.
    """

    def __init__(self, logger_name: str = "llm_security_pipeline.audit", level: int = logging.INFO):
        self._logger = logging.getLogger(logger_name)
        self._level = level

    async def log(self, event_type: str, data: dict) -> None:
        record = {"ts": time.time(), "event": event_type, "data": data}
        extra: dict[str, object] = {"audit.event": event_type}
        for key, value in data.items():
            extra[f"audit.{key}"] = (
                value if isinstance(value, (str, int, float, bool)) or value is None
                else json.dumps(value, default=str)
            )
        self._logger.log(self._level, json.dumps(record, ensure_ascii=False, default=str), extra=extra)
