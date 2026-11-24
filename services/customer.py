"""Fetch and format customer account data from the configured WordPress site."""

import os
import json
import time
import hmac
import hashlib
import base64
import httpx
from typing import Dict, Any, Optional


# Simple TTL cache: {email: (data, timestamp)}
from services.company_pack import product_settings, wording

_cache: Dict[str, tuple] = {}
_CACHE_TTL = 300  # 5 minutes


async def _fetch_from_wp(email: str) -> Optional[Dict[str, Any]]:
    """Fetch customer data from the configured WordPress endpoint.

    Uses HMAC-SHA1 signature matching HelpScout's webhook format.
    Returns parsed JSON dict or None on any failure.
    """
    url = os.getenv(product_settings()["customer_url_variable"])
    secret = os.getenv(product_settings()["customer_secret_variable"])

    if not url or not secret:
        return None

    body = json.dumps({"customer": {"email": email}})
    signature = base64.b64encode(
        hmac.new(secret.encode(), body.encode(), hashlib.sha1).digest()
    ).decode()

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                url,
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "X-HelpScout-Signature": signature,
                },
                params={"format": "json"},
            )
            response.raise_for_status()
            return response.json()

    except Exception as e:
        print(f"Customer data fetch error: {e}")
        return None


async def fetch_customer_data(email: str) -> Optional[Dict[str, Any]]:
    """Fetch customer data with TTL caching.

    Returns dict or None. Caches results for 5 minutes.
    """
    if not email:
        return None

    now = time.time()

    # Check cache
    if email in _cache:
        data, cached_at = _cache[email]
        if now - cached_at < _CACHE_TTL:
            return data

    data = await _fetch_from_wp(email)
    _cache[email] = (data, now)
    return data


def format_customer_context(data: Optional[Dict[str, Any]]) -> str:
    """Convert customer JSON to a concise text block for the LLM.

    Returns empty string if data is None or empty.
    """
    if not data:
        return ""

    # The endpoint answers an unknown address with {"html": "...No user found..."} and a
    # lookup it could not complete with {"error": ...}; neither is an account, so neither may
    # be rendered as one ("Customer type: Direct" with no orders read as a real customer).
    if "woocommerce" not in data and "licenses" not in data:
        if data.get("error"):
            return wording("lookup_error")
        if "no user found" in str(data.get("html", "")).lower():
            email = data.get("email") or "this email"
            return wording("account_miss").format(email=email)
        return ""

    parts = ["=== CUSTOMER ACCOUNT INFO ==="]

    # Customer type
    is_envato = data.get("is_envato_client", False)
    customer_type = wording("marketplace_customer") if is_envato else wording("direct_customer")
    parts.append(f"Customer type: {customer_type}")

    # Orders summary
    woo = data.get("woocommerce", {})
    order_count = woo.get("order_count", 0)
    total_spent = woo.get("total_spent", "0")
    if order_count:
        parts.append(f"Orders: {order_count} (total spent: ${total_spent})")

    # Licenses
    licenses_by_order = data.get("licenses", {})
    # PHP serializes an empty associative collection as [], not {}.
    if licenses_by_order == []:
        licenses_by_order = {}
    license_lines = []
    for order_id, order_data in licenses_by_order.items():
        for lic in order_data.get("licenses", []):
            status = lic.get("license_status", "unknown")
            product = lic.get("product_title", "Unknown Product")
            expiry = lic.get("license_expiry_date", "")
            lic_type = lic.get("license_type", "")

            details = [status]
            if lic_type:
                details.append(lic_type)
            if expiry:
                details.append(f"expires {expiry}")

            license_lines.append(f"  - {product}: {', '.join(details)}")

    if license_lines:
        parts.append("Licenses:")
        parts.extend(license_lines)

    # Active installations
    active_themes = data.get("active_themes", [])
    if active_themes:
        install_lines = []
        for theme in active_themes:
            name = theme.get("name", "Unknown")
            version = theme.get("version", "")
            site = theme.get("site_url", "")
            version_str = f" v{version}" if version else ""
            site_str = f" at {site}" if site else ""
            install_lines.append(f"  - {name}{version_str}{site_str}")

        parts.append("Active installations:")
        parts.extend(install_lines)

    parts.append("=== END CUSTOMER INFO ===")
    return "\n".join(parts)
