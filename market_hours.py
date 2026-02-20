"""
scheduler/market_hours.py
Author: Yang
Description: US equity market hours detection.
             Assumes the host machine runs on UTC. Adjust UTC offset if needed.
"""

from datetime import datetime

# US Eastern Time offset from UTC
# ET = UTC-5 (EST) or UTC-4 (EDT — daylight saving)
# Simple approximation: treat as UTC-5 year-round; for production use pytz or zoneinfo.
_ET_UTC_OFFSET_HOURS = 5


def is_weekend() -> bool:
    """Return True on Saturday (5) or Sunday (6) UTC."""
    return datetime.utcnow().weekday() >= 5


def is_market_hours() -> bool:
    """
    Return True during NYSE core session: Monday–Friday 09:30–16:00 ET.
    Uses a fixed UTC-5 offset — replace with pytz for DST accuracy.
    """
    utc_now   = datetime.utcnow()
    et_hour   = (utc_now.hour - _ET_UTC_OFFSET_HOURS) % 24
    et_minute = utc_now.minute
    weekday   = utc_now.weekday()

    if weekday >= 5:
        return False

    return (9, 30) <= (et_hour, et_minute) <= (16, 0)


def is_pre_market() -> bool:
    """Return True during pre-market session 04:00–09:29 ET."""
    utc_now   = datetime.utcnow()
    et_hour   = (utc_now.hour - _ET_UTC_OFFSET_HOURS) % 24
    et_minute = utc_now.minute
    if utc_now.weekday() >= 5:
        return False
    return (4, 0) <= (et_hour, et_minute) < (9, 30)


def seconds_until_open() -> int:
    """
    Return approximate seconds until the next NYSE open.
    Useful for sleeping the agent precisely until 09:30 ET.
    """
    utc_now    = datetime.utcnow()
    et_hour    = (utc_now.hour - _ET_UTC_OFFSET_HOURS) % 24
    et_minute  = utc_now.minute
    et_second  = utc_now.second

    # Minutes remaining in current day until 09:30 ET
    open_minutes  = 9 * 60 + 30
    current_et    = et_hour * 60 + et_minute
    remaining_sec = (open_minutes - current_et) * 60 - et_second

    # If market already open or past close, add a day
    if remaining_sec <= 0:
        remaining_sec += 24 * 3600

    return int(remaining_sec)
