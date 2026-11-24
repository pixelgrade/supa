# Contributing to Supa

Issues and pull requests are welcome. Describe the change, the behavior it affects and how you verified it. Contributions are licensed under AGPL-3.0-or-later, the same license as Supa. See [LICENSE](LICENSE).

From a fresh checkout on a supported Python 3.12 platform, run the suite with the example pack.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes -r requirements.txt
python -m pip install --no-build-isolation --no-deps .
python -m pip check
python -m pytest -q
```

Use invented people, products and reserved test domains in fixtures. Keep credentials, mailbox contents and runtime databases out of issues, tests and commits. Use the [private reporting instructions](SECURITY.md) for vulnerabilities.
