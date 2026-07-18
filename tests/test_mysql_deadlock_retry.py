"""
Unit tests for the MySQL deadlock-retry wrapper: no real MySQL needed. The
module's own docstring says the retry behavior was "verified empirically,
not just from the docs" under real cross-process contention — but no test
actually forces a real InnoDB deadlock (1213) to fire that retry path;
reliably forcing a genuine deadlock in a portable test is impractical. This
isolates the retry/backoff DECISION logic itself with a fake flaky
operation instead, which is the standard way to test retry wrappers
without depending on real database lock timing.
"""

from __future__ import annotations

import pymysql
import pytest

pytest.importorskip("aiomysql")  # mysql_stores.py only sets its module-level pymysql if aiomysql import succeeds too

from llm_security_pipeline.sessions.mysql_stores import (
    _DEADLOCK_ERRNO,
    _LOCK_WAIT_TIMEOUT_ERRNO,
    _MAX_RETRIES,
    _is_retryable,
    _with_deadlock_retry,
)


def test_is_retryable_true_for_deadlock_and_lock_timeout_errnos():
    assert _is_retryable(pymysql.err.OperationalError(_DEADLOCK_ERRNO, "Deadlock found")) is True
    assert _is_retryable(pymysql.err.OperationalError(_LOCK_WAIT_TIMEOUT_ERRNO, "Lock wait timeout")) is True


def test_is_retryable_false_for_other_operational_errors():
    assert _is_retryable(pymysql.err.OperationalError(1046, "No database selected")) is False


def test_is_retryable_false_for_non_mysql_exceptions():
    assert _is_retryable(ValueError("not a deadlock")) is False


async def test_with_deadlock_retry_retries_then_succeeds():
    attempts = []

    async def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise pymysql.err.OperationalError(_DEADLOCK_ERRNO, "Deadlock found")
        return "ok"

    result = await _with_deadlock_retry(flaky)

    assert result == "ok"
    assert len(attempts) == 3


async def test_with_deadlock_retry_gives_up_after_max_retries():
    attempts = []

    async def always_deadlocks():
        attempts.append(1)
        raise pymysql.err.OperationalError(_DEADLOCK_ERRNO, "Deadlock found")

    with pytest.raises(pymysql.err.OperationalError):
        await _with_deadlock_retry(always_deadlocks)

    assert len(attempts) == _MAX_RETRIES


async def test_with_deadlock_retry_does_not_retry_non_retryable_error():
    attempts = []

    async def raises_value_error():
        attempts.append(1)
        raise ValueError("not a deadlock")

    with pytest.raises(ValueError):
        await _with_deadlock_retry(raises_value_error)

    assert len(attempts) == 1