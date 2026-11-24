"""Inline image extraction from invented reader messages."""

import pytest

pytestmark = pytest.mark.usefixtures("inline_assets")

from services.helpscout import describe_attachments, extract_customer_attachments, extract_inline_images


_CDN = "https://assets.woodland.example/inline/1/mail/image"


def _customer(body):
    return {
        "type": "customer", "createdBy": {"type": "customer"},
        "createdAt": "2024-03-10T08:30:00Z", "body": body,
        "_embedded": {"attachments": []},
    }


def test_screenshot_word_split_around_pasted_image_remains_primary_evidence():
    # A mobile email split "screenshots" around the image, with an opaque Gmail alt.
    body = (
        'My journal is missing its last page. See s<br>'
        f'<img src="{_CDN}/602.png" alt="ii_a1b2c3d4e5f6a7b8c9" />'
        '<br><br>creenshots'
    )

    [image] = extract_customer_attachments([_customer(body)])

    assert image["is_image"] and not image["is_blank"]
    assert "primary evidence" in describe_attachments([image])


def test_large_inline_image_in_problem_description_is_not_a_signature_logo():
    body = (
        '<div>Opening a journal index shows a blank page.</div>'
        f'<img src="{_CDN}/image.png" alt="image.png" width="510" height="218" />'
        '<div>Reloading restores the index.</div>'
        '<div class="helpscout-signature">Thanks, Customer.</div>'
    )

    [image] = extract_customer_attachments([_customer(body)])

    assert image["is_image"]
    assert "signature logo" not in describe_attachments([image])


def test_signature_image_never_becomes_screenshot_from_message_wide_hint():
    body = (
        '<div>My screenshot is attached separately.</div>'
        '<div class="helpscout-signature"><div class="ex-gmail_signature">'
        f'<img src="{_CDN}/image.png" width="510" height="218" /></div></div>'
    )

    [image] = extract_inline_images({"body": body})

    assert not image["is_image"]
    assert "signature logo" in describe_attachments([image])


def test_descriptive_url_is_not_hidden_by_opaque_alt_text():
    body = f'<img src="{_CDN}/Screenshot-settings.png" alt="ii_opaque" />'

    [image] = extract_inline_images({"body": body})

    assert image["is_image"]


def test_named_signature_logo_outside_signature_container_stays_non_evidence():
    body = (
        '<div>The journal index adds an extra entry.</div>'
        f'<img src="{_CDN}/company-logo.png" alt="Company logo" width="400" height="160" />'
    )

    [image] = extract_inline_images({"body": body})

    assert not image["is_image"]


def test_ambiguous_pasted_image_is_described_as_present_without_inventing_logo():
    [image] = extract_inline_images({"body": f'<img src="{_CDN}/unknown.png" />'})

    summary = describe_attachments([image])

    assert not image["is_image"]
    assert "signature logo" not in summary
    assert "not verified" in summary


def test_thin_screenshot_is_not_dropped_as_a_social_icon():
    body = f'<img src="{_CDN}/Screenshot-toolbar.png" width="800" height="40" />'

    [image] = extract_inline_images({"body": body})

    assert image["is_image"]


def test_quoted_image_is_not_evidence_from_latest_customer_message():
    body = (
        '<div>Thanks for the previous help. My new question is about fonts.</div>'
        f'<blockquote><img src="{_CDN}/Screenshot-old.png" width="800" height="400" /></blockquote>'
    )

    assert extract_inline_images({"body": body}) == []


def test_trackers_social_icons_and_external_images_stay_excluded():
    body = (
        f'<img src="{_CDN}/tracker.png" width="1" height="1" />'
        f'<img src="{_CDN}/linkedin.png" width="16" height="16" />'
        '<img src="https://external.example/inline/Screenshot.png" width="800" height="400" />'
    )

    assert extract_inline_images({"body": body}) == []


def test_screenshot_of_logo_settings_is_not_discarded_as_a_logo():
    body = f'<img src="{_CDN}/Screenshot-logo-settings.png" alt="Screenshot of logo settings" />'

    [image] = extract_inline_images({"body": body})

    assert image["is_image"]


def test_opaque_alt_does_not_replace_image_format_from_url():
    body = f'<img src="{_CDN}/Screenshot-settings.jpeg" alt="ii_opaque" />'

    [image] = extract_inline_images({"body": body})

    assert image["mime_type"] == "image/jpeg"


def test_blank_attachment_does_not_erase_ambiguous_present_inline_image():
    images = extract_customer_attachments([{
        **_customer(f'<img src="{_CDN}/unknown.png" />'),
        "_embedded": {"attachments": [
            {"filename": "image001.png", "mimeType": "image/png", "size": 185},
        ]},
    }])

    summary = describe_attachments(images)

    assert "image purpose not verified" in summary
    assert "Every image is a blank artifact" not in summary


def test_logo_named_image_in_customer_screenshot_prose_is_not_assumed_signature():
    body = f'<div>Here is the screenshot of my broken page:</div><img src="{_CDN}/logo.png" width="510" height="218" />'
    [image] = extract_inline_images({"body": body})
    assert image["is_image"]
    assert not image["is_signature"]


def test_logo_filename_alone_is_unknown_purpose_and_still_requires_inspection():
    from services.helpscout import material_customer_images
    body = f'<div>My logo looks wrong below:</div><img src="{_CDN}/logo.png" width="400" height="160" />'
    [image] = extract_inline_images({"body": body})
    assert not image["is_signature"]
    assert not image["is_image"]
    assert material_customer_images([image]) == [image]
    assert "image purpose not verified" in describe_attachments([image])
