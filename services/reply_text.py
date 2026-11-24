"""Extract the author’s own text from a mailbox message."""

import html
import re


def own_words(text):
    """Drop quoted reply headers left in the body."""
    text = re.split(r"\bOn [^\n]{5,200}? wrote:|-{3,}\s*Original Message|\bFrom: [^\n]{0,200}? Sent: ",
                    text, maxsplit=1)[0]
    return text.strip()


# Match each language's opener with its own reply verb. Mixing them lets ordinary
# English "I am ..." consume a later French "Le ... a écrit" header and cut
# customer requirements out of flattened Help Scout email bodies.
_REPLY_HEADER_PAIRS = (
    ("On", "wrote"), ("Op", "schreef"), ("Am", "schrieb"), ("Le", "a écrit"),
    ("El", "escribió"), ("Em", "escreveu"), ("Il", "ha scritto"),
    ("Den", "skrev"), ("Dne", "napsal"), ("W dniu", "napisał"), ("Pe", "a scris"),
)
REPLY_HEADER = re.compile(
    "|".join(rf"\b{opener}\b[^\n]{{3,200}}?\b{verb}\b[^\n]{{0,160}}?:"
             for opener, verb in _REPLY_HEADER_PAIRS)
    + r"|\d{1,2}\.? \w{3,9}\.? \d{4},? (kl\.|um|à|at) \d{1,2}[:.]\d{2}[^\n]{0,80}?(skrev|schrieb|wrote|a écrit)"
    r"|-{2,}\s*(Original Message|Mensaje original|Ursprüngliche Nachricht|Message d'origine|Messaggio originale)"
    r"|_{10,}|\bFrom: [^\n]{0,200}?(Sent|Date|Gesendet|Envoyé): |\bVon: [^\n]{0,200}?Gesendet: ", re.I)
QUOTE_START = re.compile(r"(^|\s)>\s?\S")
FOOTER = re.compile(r"(🤔\s*)?How would you rate my reply\?.*|To unsubscribe from this group.*", re.I | re.S)


def customer_words(text):
    """Remove quoted history, reply headers and list footers from a message."""
    text = FOOTER.sub("", own_words(text or ""))
    header = REPLY_HEADER.search(text)
    if header:
        text = text[:header.start()]
    quote = QUOTE_START.search(text)
    if quote and text[quote.start():].count(">") >= 2:
        text = text[:quote.start()]
    return text.strip()


def body_lines(body_html):
    """A thread body as text with its line breaks kept (br, p and div end a line)."""
    text = re.sub(r"(?i)<br\s*/?>|</(p|div|li)>", "\n", body_html or "")
    text = html.unescape(re.sub(r"<[^>]+>", "", text)).replace("\xa0", " ")
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
