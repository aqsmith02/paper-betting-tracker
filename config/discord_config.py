"""
Discord Configuration

"""

import os


def get_discord_webhook(webhook_name):
    """Get Discord webhook URL from environment or local config.

    Args:
        webhook_name: Name of the Discord webhook URL (e.g., 'DISCORD_WEBHOOK')
    """
    # Try environment variable first (for GitHub Actions)
    webhook = os.environ.get(webhook_name)

    if webhook:
        return webhook

    # Fall back to local config for development
    try:
        from config.discord_config_local import DISCORD_WEBHOOK

        local_webhook = {
            "DISCORD_WEBHOOK": DISCORD_WEBHOOK,
        }
        return local_webhook.get(webhook_name)
    except ImportError:
        raise ValueError(f"{webhook_name} not found in environment or local config")


DISCORD_WEBHOOK = get_discord_webhook("DISCORD_WEBHOOK")