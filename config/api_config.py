"""
API Configuration

"""

import os


def get_api_key(key_name):
    """Get API key from environment or local config.

    Args:
        key_name: Name of the API key (e.g., 'THEODDSAPI_KEY', 'THESPORTSDB_KEY')
    """
    # Try environment variable first (for GitHub Actions)
    api_key = os.environ.get(key_name)

    if api_key:
        return api_key

    # Fall back to local config for development
    try:
        from config.api_config_local import THEODDSAPI_KEY, THESPORTSDB_KEY

        local_keys = {
            "THEODDSAPI_KEY": THEODDSAPI_KEY,
            "THESPORTSDB_KEY": THESPORTSDB_KEY,
        }
        return local_keys.get(key_name)
    except ImportError:
        raise ValueError(f"{key_name} not found in environment or local config")


THEODDSAPI_KEY = get_api_key("THEODDSAPI_KEY")
THESPORTSDB_KEY = get_api_key("THESPORTSDB_KEY")
