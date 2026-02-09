"""
Discord Configuration Example

This is a template file for local development. Copy it to create your actual config:
    cp discord_config_local.example.py discord_config_local.py

Then edit discord_config_local.py with your actual Discord webhook URL.

IMPORTANT:
- discord_config.example.py is committed to version control
- discord_config_local.py is in .gitignore and used for local development only
- In production (GitHub Actions), keys are loaded from environment variables
"""

# ============================================================================
# Discord Webhook URL (Local Development Only)
# ============================================================================

# The Discord webhook URL
DISCORD_WEBHOOK_URL = "YOUR_DISCORD_WEBHOOK_URL_HERE"

# ============================================================================
# Setup Instructions
# ============================================================================
"""
LOCAL DEVELOPMENT SETUP:
-----------------------
1. Copy this file:
   cp discord_config_local.example.py discord_config_local.py

2. Edit discord_config_local.py and replace "YOUR_DISCORD_WEBHOOK_URL_HERE" with your actual Discord webhook URL

3. Verify discord_config_local.py is in .gitignore

GITHUB ACTIONS SETUP:
--------------------
1. Go to your repository on GitHub
2. Navigate to Settings > Secrets and variables > Actions
3. Add repository secret:
   - DISCORD_WEBHOOK

4. In your workflow YAML, pass them as environment variables:
   env:
     DISCORD_WEBHOOK: ${{ secrets.DISCORD_WEBHOOK }}


USAGE:
------
In your code, import from discord_config:

    from config.discord_config import DISCORD_WEBHOOK

The discord_config.py module will automatically:
- Use environment variables in GitHub Actions
- Fall back to discord_config_local.py for local development
"""
