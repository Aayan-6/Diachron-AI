"""
Network Connectivity State Manager for SatQuery AI (ISRO PS 26167)
==================================================================
Manages online vs. air-gapped / offline detection with zero-delay caching.
Ensures the platform operates normally when connected, but seamlessly
switches to offline local catalogs, local weights, and tactical views
when completely disconnected.
"""

import os
import time
import socket
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Environment variable to explicitly force offline air-gapped operation
ENV_OFFLINE_MODE = "OFFLINE_MODE"

# Cached connectivity state: (is_online: bool, last_checked_epoch: float)
_CONNECTIVITY_CACHE = {"is_online": True, "last_checked": 0.0}
CACHE_TTL_SECONDS = 30.0
_FORCED_OFFLINE: bool = False


def is_forced_offline() -> bool:
    """Checks if offline mode is explicitly requested via env or runtime flag."""
    if _FORCED_OFFLINE:
        return True
    val = os.getenv(ENV_OFFLINE_MODE, "false").lower().strip()
    return val in ("true", "1", "yes", "on")


def set_forced_offline(enabled: bool) -> None:
    """Sets runtime forced offline toggle."""
    global _FORCED_OFFLINE
    _FORCED_OFFLINE = enabled
    _CONNECTIVITY_CACHE["last_checked"] = 0.0  # Invalidate cache


def check_internet_connection(timeout: float = 0.8) -> bool:
    """
    Checks if active internet connection is available.
    Uses cached result if within TTL to avoid blocking latency.
    """
    if is_forced_offline():
        return False

    now = time.time()
    if now - _CONNECTIVITY_CACHE["last_checked"] < CACHE_TTL_SECONDS:
        return _CONNECTIVITY_CACHE["is_online"]

    # Test DNS / socket connection to a reliable public DNS endpoint
    online = False
    try:
        # Fast DNS lookup or TCP connect to Cloudflare DNS 1.1.1.1:53
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(("1.1.1.1", 53))
        sock.close()
        online = True
    except Exception:
        online = False

    _CONNECTIVITY_CACHE["is_online"] = online
    _CONNECTIVITY_CACHE["last_checked"] = now
    return online


def is_online() -> bool:
    """Returns True if the system is currently connected to the internet."""
    return check_internet_connection()


def get_network_status() -> Dict[str, Any]:
    """Returns comprehensive network status metadata for frontend & logging."""
    online = check_internet_connection()
    return {
        "is_online": online,
        "mode": "online" if online else "offline_airgapped",
        "forced_offline": is_forced_offline(),
        "timestamp": time.time(),
        "description": "Connected to public satellite STAC catalogs & remote services"
        if online
        else "Operating in 100% offline air-gapped mode. Relying exclusively on local models and indexed imagery.",
    }
