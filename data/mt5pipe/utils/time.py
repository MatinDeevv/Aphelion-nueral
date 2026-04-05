"""UTC time helpers — all internal timestamps are UTC."""

from __future__ import annotations

import datetime as dt


def utc_now() -> dt.datetime:
    """Current UTC datetime, timezone-aware."""
    return dt.datetime.now(dt.timezone.utc)


def utc_ms() -> int:
    """Current UTC as milliseconds since epoch."""
    return int(utc_now().timestamp() * 1000)


def ms_to_utc(ms: int) -> dt.datetime:
    """Convert epoch-milliseconds to UTC datetime."""
    return dt.datetime.fromtimestamp(ms / 1000.0, tz=dt.timezone.utc)


def dt_to_utc(d: dt.datetime) -> dt.datetime:
    """Ensure a datetime is UTC. Naive datetimes assumed UTC."""
    if d.tzinfo is None:
        return d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc)


def date_range(start: dt.date, end: dt.date) -> list[dt.date]:
    """Inclusive date range."""
    days: list[dt.date] = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += dt.timedelta(days=1)
    return days


def floor_dt(d: dt.datetime, seconds: int) -> dt.datetime:
    """Floor datetime to the nearest multiple of `seconds`."""
    ts = int(d.timestamp())
    floored = ts - (ts % seconds)
    return dt.datetime.fromtimestamp(floored, tz=dt.timezone.utc)
