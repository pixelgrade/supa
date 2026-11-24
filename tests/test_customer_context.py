from services.customer import format_customer_context
from services.company_pack import wording


def test_account_context_distinguishes_an_unknown_recipient_from_a_failed_lookup():
    assert format_customer_context(None) == ""
    assert format_customer_context({}) == ""
    assert format_customer_context({"error": "Lookup unavailable"}) == wording("lookup_error")
    email = "reader@woodland.example"
    assert format_customer_context({"email": email, "html": "No user found"}) == wording("account_miss").format(email=email)


def test_known_account_preserves_its_orders_and_product_information():
    text = format_customer_context({"email": "reader@woodland.example", "is_envato_client": False,
        "woocommerce": {"order_count": 2, "total_spent": 14},
        "licenses": {"31": {"licenses": [{"product_title": "Field Journal", "license_status": "valid"}]}}})
    assert wording("direct_customer") in text
    assert "Orders: 2" in text and "Field Journal: valid" in text


def test_empty_license_array_is_valid_and_leaves_no_license_rows():
    text = format_customer_context({"licenses": [], "woocommerce": {"order_count": 0}})
    assert wording("direct_customer") in text and "Orders:" not in text
    assert "Licenses:" not in text
