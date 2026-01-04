"""
API Configuration Example

This is a template file for local development. Copy it to create your actual config:
    cp api_config.example.py api_config_local.py

Then edit api_config_local.py with your actual API keys.

IMPORTANT: 
- api_config.example.py is committed to version control
- api_config_local.py is in .gitignore and used for local development only
- In production (GitHub Actions), keys are loaded from environment variables
"""

# ============================================================================
# API Keys (Local Development Only)
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
LOCAL DEVELOPMENT SETUP:
-----------------------
1. Copy this file:
   cp api_config.example.py api_config_local.py

2. Edit api_config_local.py and replace "YOUR_KEY_HERE" with your actual keys

3. Verify api_config_local.py is in .gitignore


GITHUB ACTIONS SETUP:
--------------------
1. Go to your repository on GitHub
2. Navigate to Settings > Secrets and variables > Actions
3. Add repository secrets:
   - THE_ODDS_API_KEY
   - THE_SPORTS_DB_API_KEY

4. In your workflow YAML, pass them as environment variables:
   env:
     THE_ODDS_API_KEY: ${{ secrets.THE_ODDS_API_KEY }}
     THE_SPORTS_DB_API_KEY: ${{ secrets.THE_SPORTS_DB_API_KEY }}


USAGE:
------
In your code, import from api_config:

    from config.api_config import THE_ODDS_API_KEY, THE_SPORTS_DB_API_KEY

The api_config.py module will automatically:
- Use environment variables in GitHub Actions
- Fall back to api_config_local.py for local development
"""