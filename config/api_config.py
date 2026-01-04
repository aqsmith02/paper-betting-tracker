"""
API Configuration

"""

import os

def get_api_key(key_name):
    """Get API key from environment or local config.
    
    Args:
        key_name: Name of the API key (e.g., 'THE_ODDS_API_KEY', 'THE_SPORTS_DB_API_KEY')
    """
    # Try environment variable first (for GitHub Actions)
    api_key = os.environ.get(key_name)
    
    if api_key:
        return api_key
    
    # Fall back to local config for development
    try:
        from config.api_config_local import THE_ODDS_API_KEY, THE_SPORTS_DB_API_KEY
        local_keys = {
            'THE_ODDS_API_KEY': THE_ODDS_API_KEY,
            'THE_SPORTS_DB_API_KEY': THE_SPORTS_DB_API_KEY
        }
        return local_keys.get(key_name)
    except ImportError:
        raise ValueError(f"{key_name} not found in environment or local config")

THE_ODDS_API_KEY = get_api_key('THE_ODDS_API_KEY')
THE_SPORTS_DB_API_KEY = get_api_key('THE_SPORTS_DB_API_KEY')