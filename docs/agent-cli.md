# Using the Supa CLI

The CLI connects to the Supa server over HTTPS. The server connects to the Inbox using its own credentials. Give each person or client a separate Supa identity with the scopes and conversations it needs. The installation guide walks through setting up both services and processing an invented ticket with sending disabled. [Installation guide](../README.md)

## Check your connection

Use `supa configure --url https://supa.woodland.example --token-stdin` to store an issued token on the client. Supply it on standard input. For an ephemeral process, the client also accepts `SUPA_URL` and `SUPA_TOKEN`. Keep the token out of shell arguments, logs and version control.

```sh
supa health
supa capabilities
supa conversation get 21
supa conversation timeline 21
```

`capabilities` reports the server commands and the token’s scopes. Its `client_missing` list identifies commands advertised by the server that the installed client cannot run. Server extensions can add commands. A conversation-scoped token can access only the conversations declared for it. An unrestricted read identity can use `supa desk` and `supa mailbox operations` for a mailbox-wide summary.

Use the resource ID returned by the API with CLI commands. When you have only a displayed ticket number, search for it and check the ID and number on the returned conversation. The timeline distinguishes customer messages, published replies, internal notes and unsent drafts.

## Process one fictional ticket

With the Woodland sample from the installation guide running locally, choose an idempotency key for this one request. Submit the task, then use the run ID from its response to watch progress.

```sh
supa ticket process 21 --idempotency-key woodland-proof-1
supa run watch RUN_ID
supa draft get 21
```

The processor classifies the conversation, gathers evidence, drafts and invokes the separate send-confidence judge. The task defaults to a draft-only ceiling. A token without send scope cannot authorize sending. A completed run is distinct from a published reply. Inspect the stored draft or published thread to verify what happened.

## Review and edit a draft

`supa draft get ID` returns the newest draft and its version. For an edit, give `supa draft save --help` the same client version you use for the write, and supply the returned version as the expected version. If the draft or customer thread has changed, read the conversation again before acting. The expected thread for a reply is the latest actionable customer thread. Reuse an idempotency key only when retrying the exact same operation and payload. Use `supa operation get --help` to inspect the durable result of an uncertain request.

Publishing and scheduling a reply are separate operations with their own authorization. A scheduled send rechecks ownership, token scope, the customer thread, the draft, conversation status and intervening notes or replies. The worker holds a changed conversation for review.

A copied address must belong to the customer or have been supplied in a published customer message on this conversation. Supa ignores quoted history, internal notes and agent drafts when deciding whether an address was supplied. The installation’s own mailboxes are excluded. An installed recipient check may also refuse an address. A failed recipient check holds the write.

## Records and events

Use `supa note --help` for internal notes and `supa request --help` for requests that the owner can resolve. `supa conversation state --help` lists the state fields supported by the installed client. Use stage values defined by the server.

For an event consumer, use a stable consumer name, process the returned event durably, then acknowledge its token. An unacknowledged event may be replayed. See `supa events next --help` and `supa events ack --help` for the exact arguments.
