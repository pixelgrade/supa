"""Stable reply spacing and strict comparison of stored email HTML."""
import re
from html.parser import HTMLParser


_BLOCK_SPACER = "<p>&nbsp;</p>"
_BLANK_PARAGRAPH = r"<p\s*>(?:\s|&nbsp;|&#0*160;|&#x0*a0;|<br\s*/?>)*</p\s*>"
_SPACER_RUN = re.compile(rf"({_BLANK_PARAGRAPH}(?:\s*{_BLANK_PARAGRAPH})*)", re.IGNORECASE)
_BLOCK_BOUNDARY = re.compile(
    r"(</(?:p|ul|ol|h[1-6]|blockquote)>)\s*(?=<(?:p|ul|ol|h[1-6]|blockquote)\b)",
    re.IGNORECASE,
)


class _ProtectedEmailSections(HTMLParser):
    """Find original source ranges; never reserialize code or nested quoted threads."""
    def __init__(self, html: str):
        super().__init__(convert_charrefs=False)
        self.html = html
        self.line_offsets = [0, *(match.end() for match in re.finditer("\n", html))]
        self.stack = []
        self.start = 0
        self.ranges = []
        self.feed(html)
        self.close()
        if self.stack:
            self.ranges.append((self.start, len(html)))

    def _offset(self):
        line, column = self.getpos()
        return self.line_offsets[line - 1] + column

    def handle_starttag(self, tag, attrs):
        if tag in {"pre", "code", "table", "blockquote"}:
            if not self.stack:
                self.start = self._offset()
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag not in self.stack:
            return
        self.stack = self.stack[:len(self.stack) - 1 - self.stack[::-1].index(tag)]
        if not self.stack:
            self.ranges.append((self.start, self.html.index(">", self._offset()) + 1))

    def handle_comment(self, data):
        if not self.stack:
            start = self._offset()
            self.ranges.append((start, self.html.index("-->", start) + 3))


def ensure_email_paragraph_spacing(html: str) -> str:
    """Use one blank paragraph between reply blocks, independent of formatting passes.

    Help Scout strips paragraph margins. Collapse existing blank-paragraph runs and add
    missing spacers locally; an existing spacer elsewhere must not suppress that work.
    Preserve code, tables, quoted HTML and br-style messages without reserializing them.
    """
    if not html or "<p" not in html.lower():
        return html
    try:
        protected_ranges = _ProtectedEmailSections(html).ranges
    except (ValueError, AssertionError):
        # Formatting must not block delivery or rewrite uncertain code/quote boundaries.
        return html

    def format_prose(prose: str) -> str:
        # Splitting around existing spacer runs prevents inserting a new spacer on each
        # side of one, which previously grew a gap from one to three to seven blank lines.
        return "".join(
            _BLOCK_SPACER if index % 2 else _BLOCK_BOUNDARY.sub(r"\1" + _BLOCK_SPACER, part)
            for index, part in enumerate(_SPACER_RUN.split(prose))
        )

    parts = []
    end = 0
    for start, stop in protected_ranges:
        parts.extend([format_prose(html[end:start]), html[start:stop]])
        end = stop
    parts.append(format_prose(html[end:]))
    return "".join(parts)


class _EmailTokens(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tokens = []

    def handle_starttag(self, tag, attrs):
        # With duplicate attributes the first value can decide browser behaviour, so their
        # order is significant. Unique attributes may be reordered by the upstream API.
        if len({name for name, _ in attrs}) == len(attrs):
            attrs = sorted(attrs, key=lambda a: (a[0], a[1] or "", a[1] is None))
        self.tokens.append(("start", tag, tuple(attrs)))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        self.tokens.append(("end", tag))

    def handle_data(self, data):
        if self.tokens and self.tokens[-1][0] == "data":
            self.tokens[-1] = ("data", self.tokens[-1][1] + data)
        else:
            self.tokens.append(("data", data))

    def handle_comment(self, data):
        self.tokens.append(("comment", data))

    def handle_decl(self, decl):
        self.tokens.append(("decl", decl))

    def handle_pi(self, data):
        self.tokens.append(("pi", data))

    def unknown_decl(self, data):
        self.tokens.append(("unknown", data))


def email_html_equivalent(left: str, right: str) -> bool:
    """Decode entities once in text/attributes, preserving structure and exact whitespace.

    Unescaping entire strings would equate literal text '&lt;b&gt;' with an actual <b> tag.
    Token comparison also retains comments and every attribute, unlike plain-text comparison.
    """
    try:
        a, b = _EmailTokens(), _EmailTokens()
        a.feed(left)
        a.close()
        b.feed(right)
        b.close()
        return a.tokens == b.tokens
    except (ValueError, AssertionError):
        return False
