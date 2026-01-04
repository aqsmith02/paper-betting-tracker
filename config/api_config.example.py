"""
API Configuration Example

This is a template file. Copy it to create your actual config:
    cp api_config.example.py api_config.py

Then edit api_config.py with your actual API keys.

IMPORTANT: 
- api_config.example.py is committed to version control
- api_config.py is in .gitignore
"""

import os

# ============================================================================
# API Keys
# ============================================================================

# The Odds API key
# Get your key at: https://the-odds-api.com/
THE_ODDS_API_KEY = "YOUR_KEY_HERE"

# TheSportsDB API key
# Get your key at: https://www.thesportsdb.com/
THE_SPORTS_DB_API_KEY = "YOUR_KEY_HERE"


# ============================================================================
# Setup Instructions
# ============================================================================
"""
SETUP STEPS:
------------
1. Copy this file:
   cp api_config.example.py api_config.py

2. Edit api_config.py and replace "YOUR_KEY_HERE" with your actual keys

3. Verify api_config.py is in .gitignore


USAGE:
------
In your code, import from api_config (not api_config.example):

    from api_config import THE_ODDS_API_KEY, THESPORTSDB_API_KEY
"""