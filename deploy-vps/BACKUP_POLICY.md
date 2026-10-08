# AS-14: verified daily retention

This documents the implementation; actual VPS/service results belong in
`RELEASE_READINESS.md`. Source preparation and isolated tests alone do not prove
that the installed systemd service has run.

`ops_backup.sh` remains the existing service entry point. `ops_backup_policy.py`
holds the same `backup.lock` during all stages and rotation. The default operation
ID is one UTC calendar day. Repeating that operation verifies the same completed
set; it does not create another copy. For a deliberately fresh deployment copy,
use a new explicit operation ID:

```sh
/bin/sh /home/arborscan/ops-tools/ops_backup.sh --operation-id deploy-YYYYMMDDTHHMMSSZ
```

Interrupted runs keep `active-backup.json` and attempt folders in `.staging`.
The next invocation finishes that operation even across midnight. An interrupted
attempt is never published. A crash after publication resumes verification and
rotation from the covered operation marker, without a second completed set.
Failed attempts are retained privately for diagnosis; they do not count as
successful copies and the free-space guard bounds further creation.

`last-success.json` is written atomically **after verified publication/rotation
and before removing the active-operation marker**. Repeating an already completed
operation verifies its durable set and repairs missing or stale success metadata;
it never needs another snapshot. `completed_at_utc` comes from the existing
COMPLETE file timestamp, not from the time of that retry. For the same manifest
and operation, an earlier valid `verified_at_utc` and rotation history are retained.
If no trustworthy prior record exists, the verification-time fallback is the
actual completion time. The separate `last_integrity_check_at_utc` records the
current recheck, and `retained_verified_sets` records the verified count after
rotation. Rechecking old data therefore does not make its backup age look fresh.
Success metadata is outside the snapshot manifest; this change does not modify
the native PostgreSQL dump or stored application data.

The create stage reads application tables/Storage using the existing bounded
read retries, checks an offline file restore and report links, records v3/v4/worker
and every real Compose source, copies local model/runtime files and configuration,
and performs the existing native PostgreSQL dump. A missing Compose source fails
the stage. The new outer manifest includes the native `postgres/SHA256SUMS`.
Parent COMPLETE is published only after full outer/native checks and immutable
runtime archive checks, then atomic rename from the staging directory.

The runtime index is private and is not a claim that a Docker image store alone
is recoverable. Prepare immutable archives and validate their exact image IDs;
then configure `/home/arborscan/ops-runtime/RUNTIME_INDEX.json` (mode600):

```json
{"format":1,"images":{"sha256:EXACT_IMAGE_ID":[{"path":"/private/immutable/archive","sha256":"SHA256_OF_ACTUAL_CONTENT"}]}}
```

Include all required base/overlay parts and their manifests. New snapshots select the actual APIv3, APIv4
and worker mappings, recording an additive format-1 `services` list. Legacy sets
without this list remain v4/worker snapshots; they do not prove auth-image recovery.
The selected mappings are copied into hash-covered `RUNTIME_DEPENDENCIES.json`.
Every referenced file is read and verified before publication and when a set is
used as the new usable replacement. The source archives remain outside ordinary
daily rotation. Any runtime/Compose reference into a retained daily set protects
that set; no image archive or related manifest is independently deleted.

The retention budget is **14 verified COMPLETE sets**. Legacy pre-native archives
may be inventoried, but are labelled separately and never substitute for a new
full native/runtime replacement. Corrupt/incomplete sets are retained, diagnosed,
and excluded from the successful count. Protected/manual copies count toward the
budget; when too few eligible automatic sets remain, the procedure refuses with
`rotation_blocked_all_remaining_sets_protected` before creating another copy.
It does not silently change the budget or remove protected data.

Only exact timestamp directories immediately under the intended private root
are rotation candidates. Symlinks and path traversal are refused. `PINNED`,
`DO_NOT_ROTATE` and `ROLLBACK_PIN` protect manual/control sets. Current Docker
Compose sources/mounts and references from retained metadata are re-read for the
plan. The live `20260926T202023Z/reliability.yml` dependency is consequently
protected. Unreadable retained dependency metadata blocks deletion.

Legacy sets have no trustworthy automatic marker. They remain protected until
their exact manifest hash is explicitly adopted from an **actually observed**
automatic run. Names resembling a scheduled time are not evidence. Adoption does
not rewrite the original directory or its SHA256SUMS:

```json
{"format":1,"sets":[{"name":"YYYYMMDDTHHMMSSZ","manifest_sha256":"EXACT_ORIGINAL_MANIFEST_SHA256","source":"observed_automatic_run","evidence_reference":"actual service journal or previously recorded observed run"}]}
```

```sh
python3 /home/arborscan/ops-tools/ops_backup_policy.py adopt --evidence /private/adoption-evidence.json
python3 /home/arborscan/ops-tools/ops_backup_policy.py dry-run
```

Before creation and again before deletion, a private `rotation-dry-run.json`
lists candidates and reasons. Creation uses the existing20GiB free-space floor.
With insufficient space this implementation **refuses pre-deletion**: a historical
external receipt is not current proof that bytes are recoverable from the receiver.
It keeps every existing copy and reports
`insufficient_disk_no_fresh_external_verification_for_predelete`. Optional operator
disk provisioning is safer than deleting the only recovery copy. A failed external
transfer therefore cannot cause pre-deletion. Ordinary rotation with ample space
occurs only after the new complete/native/runtime set has been verified.

`ops_status.py` reports integrity-based native and runtime-contract freshness
separately. COMPLETE alone is insufficient. Verified external receipts are shown
as historical confirmation ages, explicitly **not** current receiver reachability.
No notifications are sent.

`tests_v4/test_ops_backup_policy.py` exercises full14в†’newв†’14, repeat, stage failure,
interruption before/after publication, next-day resume, low disk, corrupt content,
concurrent locking, pinned/manual/live sources, runtime base dependencies, missing
runtime archives, adoption, path traversal/symlinks and freshness diagnostics in
isolated temporary fixtures. Linux fcntl/shell execution and real service/timer
observations must be recorded separately from Windows tests.

The final policy suite contains **27 tests**, including interrupted success-record
publication followed by a repair without another snapshot/deletion, stale-record
repair, preserved completion/verification timestamps and rotation history, and
rejection of a future claimed verification time. On 07.10.2026 these passed on
Windows (25.642s) and VPS Linux (34.149s). The real automatic systemd run on that
date used the preceding policy source; the later metadata fix is checked by a
separate `verified_existing` operation, not reported as another systemd run.
Exact installed hashes, preimages and actual VPS outcomes are in
`evidence/release-readiness/backup-policy-idempotency-fix.json` and
`evidence/release-readiness/service-result.json`.

On 07.10.2026 the metadata policy was installed under `backup.lock` with the
service inactive; its previous source is preserved privately. The actual
`daily-2026-10-06` replay returned `verified_existing` with zero snapshot calls,
zero deletions and **14в†’14** completed sets. Every set's manifest, COMPLETE
timestamp, file count and size, plus staging files, remained unchanged. The
03:29:58 UTC completion marker and original 03:34:25 UTC verification time were
preserved; only the separate 07:01:09 UTC integrity recheck was recorded. Actual
source/preimage hashes were rechecked and v3/v4/worker remained running and
healthy with their preceding image IDs and start times. See
`evidence/release-readiness/backup-policy-existing-replay.json`. This was not a
new full backup, PostgreSQL restore or systemd-service execution.


## Windows completion, 08.10.2026

`ops_pull_backups.py` uses only existing `D:\ArborScanBackups`. Besides the
covered snapshot it transfers unique runtime archives to `runtime-assets` under
the same root, verifies SHA, then writes the existing format-1
`RELOCATED_RUNTIME.json`. A set plus this shared directory is a recoverable
package; the timestamp directory alone is not an independent full package.
Original covered manifests and source paths are not rewritten. A matching
previous recovery archive can be reused by hard link, preserving the canonical
bytes even after its old directory is retired. Interrupted downloads remain
`.downloading`; checksum failures cannot publish relocation or authorize rotation.

`ops_windows_retention.py` plans then applies the existing budget of 14 verified
COMPLETE sets only after an actual native PostgreSQL manifest/dump and all three
service/runtime SHA verifications. It rechecks before each bounded deletion.
Pins, live Compose paths, retained runtime bases and manual/unadopted historical
sets remain protected; shared image archives are never collected by this tool.
Older observed automatic sets require explicit byte-SHA-bound adoption in private
`WINDOWS_RETENTION_POLICY.private.json`, with the original successful-run evidence.
A timestamp by itself is not adoption. Insufficient eligible sets refuse rotation;
no shortening of the policy or deleting historical manual sets is implied.
Windows junction/reparse root/parents are rejected before resolving paths.

The pull holds its existing `pull.lock` across transfers and rotation. The
installed Scheduled Task points to permanent `D:\ArborScanBackups\tools`, not
a transient worktree. Its future schedule requires an enabled/logged-on Windows
PC and available SSH. A manual success is distinct from a future calendar run.
Actual run/restore/rotation receipts: [TECHNICAL_RELEASE_COMPLETION.md](TECHNICAL_RELEASE_COMPLETION.md).
