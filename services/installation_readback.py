"""Check explicit identity settings against the values consumed by this process."""

import os
from pathlib import Path

from supa_cli.installation import waiting_on_choices


def explicit_settings_match() -> bool:
    from agents import codex_cli, provider_alerts
    from api.agent import _waiting_on_values
    from services import company_pack, conversation_gateway, deep_pass, installation
    from services import helpscout, live_site_evidence, mailbox_provider, runtime_watchdog, write_canary

    env = os.environ
    required = (
        "SUPA_PACK", "SUPA_VENDOR_INBOX_URL", "SUPA_OPERATIONS_ORIGIN",
        "SUPA_CODEX_DIRECTORY", "SUPA_OWN_DOMAINS", "SUPA_CANARY_MAILBOX_ID",
        "SUPA_CANARY_CUSTOMER_ID", "SUPA_CANARY_EMAIL", "SUPA_ALERT_EMAIL_TO",
        "SUPA_OWNER_ALIAS", "LAUNCH_AGENT_LABEL", "SUPA_INLINE_IMAGE_HOSTS",
    )
    if any(not env.get(key) for key in required):
        return False
    domains = tuple(v.strip() for v in env["SUPA_OWN_DOMAINS"].split(",") if v.strip())
    return all((
        company_pack.PACK_DIR == company_pack.PROJECT_ROOT / Path(env["SUPA_PACK"]).expanduser(),
        mailbox_provider.VENDOR_INBOX == env["SUPA_VENDOR_INBOX_URL"],
        installation.OPERATIONS_ORIGIN == env["SUPA_OPERATIONS_ORIGIN"],
        codex_cli.DEFAULT_WORKING_DIRECTORY == env["SUPA_CODEX_DIRECTORY"],
        conversation_gateway._OWN_CC_DOMAINS == domains,
        deep_pass._OWN_DOMAINS == (*domains, "helpscout.net"),
        helpscout.HELPSCOUT_INLINE_IMAGE_HOSTS == tuple(v.strip() for v in env["SUPA_INLINE_IMAGE_HOSTS"].split(",") if v.strip()),
        all(d in live_site_evidence.NON_CUSTOMER_DOMAINS for d in domains),
        write_canary.MAILBOX_ID == int(env["SUPA_CANARY_MAILBOX_ID"]),
        write_canary.CUSTOMER_ID == int(env["SUPA_CANARY_CUSTOMER_ID"]),
        installation.CANARY_EMAIL == env["SUPA_CANARY_EMAIL"],
        provider_alerts._email_recipient() == env["SUPA_ALERT_EMAIL_TO"],
        _waiting_on_values == waiting_on_choices(),
        env["SUPA_OWNER_ALIAS"] in _waiting_on_values,
        runtime_watchdog.load_watchdog_config().app_launch_agent_label == env["LAUNCH_AGENT_LABEL"],
    ))
