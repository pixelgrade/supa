# Supa

Supa is a self-hosted support agent. It classifies incoming conversations, searches product documentation and earlier answers, and drafts a reply. A second model judges the draft. The default delivery mode saves replies for review. Supa is the agent half of a pair with [Inbox, the support desk](https://github.com/pixelgrade/inbox). Pixelgrade provides Supa as is and offers paid setup on request.

The included Woodland Press pack describes a fictional product called Field Journal. Use it with a fresh local Inbox to try the complete path from a reader’s question to a saved draft.

## What you need

Use CPython 3.12 on macOS 12 or newer (Apple silicon or Intel), or x86-64 Linux with glibc 2.28 or newer, SQLite with FTS5, Git, OpenSSL with `-addext`, and space for the Python environments and embedding model. The first install needs Internet access. Supa downloads the `all-MiniLM-L6-v2` embedding model on first use.

Generation and scoring use local command-line clients with access to their model services. Install Claude Code and authenticate by launching `claude` as the account that will run Supa. Check that this account can use the configured Claude models. Install and authenticate the Codex CLI with `codex login` to enable its fallback. The default writer tries Opus, Sonnet and then Codex. The judge defaults to Sonnet. An API key alone cannot start these subprocesses. Tickets and retrieved context are supplied to the configured model service.

## Start a fictional desk

Run these commands in the same shell, starting in an empty directory. The example uses two loopback listeners and a new database. Keep this directory private. The example needs no mail transport.

```sh
umask 077
export LAB="$PWD"
mkdir -m 700 private
git clone https://github.com/pixelgrade/inbox inbox
git clone https://github.com/pixelgrade/supa supa

cd "$LAB/inbox"
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.txt
export INBOX_DB="$LAB/private/inbox.db"
export INBOX_BASE_URL=https://127.0.0.1:8926
export INBOX_COMPANY_NAME='Woodland Press'
export INBOX_SUPPORT_ADDRESS='support@woodland.example'
export INBOX_INTAKE_MAILBOX=1 INBOX_AGENT_ID=1 INBOX_AGENT_NAME='Woodland Support'
export INBOX_INITIAL_AUTHORITY=standby INBOX_OPERATOR_WRITES=1
export INBOX_CLIENT_ID=woodland
export INBOX_CLIENT_SECRET="$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')"
.venv/bin/python -m workbench.init_mailbox examples/journal.json
.venv/bin/python writer_control.py live --from standby --native-company "$INBOX_COMPANY_NAME"
openssl req -x509 -newkey rsa:2048 -nodes -days 7 -subj /CN=localhost \
  -addext 'subjectAltName=DNS:localhost,IP:127.0.0.1' \
  -keyout "$LAB/private/inbox.key" -out "$LAB/private/inbox.crt"
```

Add one fictional incoming question to this new desk. This uses the Inbox import routine and does not send mail.

```sh
.venv/bin/python - <<'PY'
from contextlib import closing
from datetime import datetime, timezone
import store

now = datetime.now(timezone.utc).isoformat(timespec="seconds")
reader = {"id": 7, "first": "Rowan", "last": "Reed", "email": "reader@woodland.example"}
question = {"id": 31, "type": "customer", "state": "published", "createdAt": now,
            "createdBy": reader, "source": {"type": "email"},
            "body": "How do I open a saved Field Journal notebook?"}
conversation = {"id": 21, "number": 21, "mailboxId": 1, "state": "published",
                "status": "active", "type": "email", "subject": "Open a Field Journal notebook",
                "primaryCustomer": reader, "createdAt": now}
with closing(store.connect()) as db, store.transaction(db):
    store.import_conversation(db, conversation, [question])
PY
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8926 \
  --ssl-keyfile "$LAB/private/inbox.key" --ssl-certfile "$LAB/private/inbox.crt" \
  >"$LAB/private/inbox.log" 2>&1 &
INBOX_PID=$!
```

## Prepare a draft

Install the agent and its CLI from the second checkout. Supa uses the same client ID and secret as the Inbox. Its CLI has a separate token restricted to the fictional conversation and the read, task and draft permissions.

```sh
cd "$LAB/supa"
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes -r requirements.txt
python -m pip install --no-build-isolation --no-deps .
python -m pip check
export SUPA_MAILBOX_ORIGIN="$INBOX_BASE_URL"
export SUPA_MAILBOX_CLIENT_ID="$INBOX_CLIENT_ID"
export SUPA_MAILBOX_CLIENT_SECRET="$INBOX_CLIENT_SECRET"
cat "$(python -m certifi)" "$LAB/private/inbox.crt" > "$LAB/private/trust.pem"
export SSL_CERT_FILE="$LAB/private/trust.pem"
export SUPA_URL=http://127.0.0.1:8100
export SUPA_PACK=packs/woodland SUPA_SUPPORT_MODE=draft_only
export SUPA_DB_PATH="$LAB/private/supa.db"
export SUPA_AGENT_TOKENS_FILE="$LAB/private/agent-tokens.json"
export ADMIN_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python scripts/manage_agent_tokens.py create --name woodland --scopes read,task,draft \
  --conversations 21 > "$LAB/private/woodland-token.json"
export SUPA_TOKEN="$(python -c 'import json,os; print(json.load(open(os.environ["LAB"]+"/private/woodland-token.json"))["token"])')"
python -m uvicorn main:app --host 127.0.0.1 --port 8100 --lifespan off \
  >"$LAB/private/supa.log" 2>&1 &
SUPA_PID=$!
```

The example leaves background workers off to process this ticket once on request. When both listeners are ready, run the commands below. The processing command returns a run ID. After the run finishes, `draft get` shows the stored reply and `timeline` shows its draft state.

```sh
supa capabilities
supa conversation get 21
supa ticket process 21 --idempotency-key woodland-first-question > "$LAB/private/run.json"
RUN_ID="$(python -c 'import json,os; print(json.load(open(os.environ["LAB"]+"/private/run.json"))["result"]["run_id"])')"
supa run watch "$RUN_ID"
supa draft get 21
supa conversation timeline 21
```

Open the agent dashboard at `http://127.0.0.1:8100/`. Stop these two local servers with `kill "$SUPA_PID" "$INBOX_PID"`. For subsequent starts, keep the same configuration for both servers and store the shared secret in a private service environment file. Initialize a mailbox only once.

## Run your own installation

Point `SUPA_MAILBOX_ORIGIN` at your own Inbox HTTPS origin. Set its separate client ID and secret, an admin key, the audit database path and the scoped token registry in a private environment file. Run Supa as a dedicated account with access to its model clients and credentials. The vector index lives in `db/vectors.db` inside the checkout. The service account needs a writable `db/` and a persistent home for its model cache and authentication. Keep the environment, token files and databases private and include them in your backups.

Adapt [the systemd unit](examples/supa.service) to your paths. Its normal application startup enables recovery and mailbox workers. A reverse proxy can provide HTTPS for a networked dashboard. Keep `SUPA_SUPPORT_MODE=draft_only` for human review. Replace `SUPA_PACK` with the directory containing your product settings, prompts, rules, voice, repository catalog and knowledge files. Optional integrations load at startup from the comma-separated Python module names in `SUPA_EXTENSIONS`.

The dashboard’s source link defaults to [the corresponding project repository](https://github.com/pixelgrade/supa). Set `SUPA_SOURCE_URL` to the corresponding source for the version you serve, including your modifications, so people interacting with it can obtain that source under AGPL section 13. See [LICENSE](LICENSE), [NOTICE](NOTICE) and [the CLI guide](docs/agent-cli.md).
