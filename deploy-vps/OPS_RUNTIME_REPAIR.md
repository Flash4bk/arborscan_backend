# AS-14: immutable runtime closure repair and failed-stage quarantine

These tools prepare a reviewed repair; they do not authorize production changes.
Do not start a worker, train, prune Docker, remove COMPLETE sets, overwrite live
data or change an API as part of this procedure. User approval must name exact
failed staging trees before cleanup. Keep private records outside Git.

## Recorded verification: 17 November 2026

The separately authorized 07400c0c tools/index repair was installed on 12 November.
The owner's actual systemd start on 17 November finished with `Result=success`,
`ExecMainStatus=0`, from 12:45:24 to 13:22:26 UTC. It verified/replayed the existing
`20261117T033225Z` operation, whose data capture completed at 03:45:47 UTC;
13:22:26 is the integrity check time, not a second database snapshot. Existing
retention removed `20261013T032406Z`. The timer is active/waiting for 18 November.

The same snapshot was actually restored on 17 November in a fresh, network-none,
resource-limited PostgreSQL container: all 15 public tables matched, Storage
inventory and the three migration versions matched. Raw Storage/runtime files
also matched bytes and uid/gid/mode/mtime/xattrs. Only the newly created test
container and two named volumes were removed. The seven production container
identities and snapshot identity remained unchanged. Raw helpers mounted the
snapshot directory read-only, including its private files; the test database
did not receive production credentials. No values of private files were printed.
This is an isolated restore on the existing VPS, not a new-host recovery drill.

The 37-minute service duration exposed repeated full scans of unchanged sets.
The candidate `_VerifiedSetCache` retains the first complete SHA/native/raw/runtime
verification and is scoped to one locked Linux `run_backup` invocation. Every hit
checks exact membership, inode, Linux ctime, mtime, size, mode and ownership before
and after use; external runtime checks remain separate. Windows/other platforms
and unpublished stages keep full reads. Destructive candidates always receive
fresh, uncached full verification. No success receipt is trusted as a cache.

Checks of this candidate: Windows pytest **141 passed / 3 skipped** (two skips
need Linux ctime; one existing platform skip). On actual Linux/Python 3.12.3,
the stdlib adapter ran the unchanged 21 test cases plus four separate adversarial
ctime/hardlink cases: **21 + 4 passed**, no skips. This adapter run is not a native
pytest invocation. Fixture storage full-SHA reads fell from 44 to 12 across two
replays; production duration of the new candidate is not established by that
fixture. Installation/replay evidence must be recorded separately below.

## Evidence and source identity

Record the actual service/timer result, free bytes, installed tool hashes,
configured index hash and the seven captured Docker image IDs. A missing image
mapping is distinct from disk pressure. A created manual-worker can be an
intentional stopped state. Do not infer it is broken or start it automatically.

The code bundle must come from one reviewed Git commit. It includes the current
`ops_backup.sh`, dispatcher, backup policy, selfhost backup, runtime contract,
preflight and repair modules, plus unchanged `ops_verify_restore.py`. Preserve
installed source/config/index bytes before atomic replacement. Shell entrypoints
need LF. New tool files alone do not switch the private dispatcher config.

## Runtime archive candidate

`ops_selfhost_runtime_preflight.preflight` checks actual captured services before
database/files copy. It intentionally reports `integrity_verified: false`;
presence is not full archive integrity. Normal publication still hashes closure.

When a full image archive is unavailable but a verified local base exists,
`ops_image_delta.py apply-member` reconstructs a full, independently usable tar
from an opaque member delta. It verifies the base hash, exact target member set,
every member hash, payload budget and pinned classic/OCI image closure. Keep
partials on refusal; do not advertise COMPLETE. Example with values from the
private reviewed receipt:

```sh
python3 -B ops_image_delta.py apply-member "$base_archive" "$member_delta" "$new_partial_archive" \
  --image-id "$actual_image_id" --base-sha256 "$base_sha256" \
  --expected-archive-sha256 "$expected_archive_sha256"
```

After exact verification, rename only inside the new private candidate location.
`ops_runtime_closure_repair.py` creates a **new** index directory, retains prior
image mappings, verifies all old assets and refuses a source-index race. It does
not switch config or export/start Docker. Its `--inventory` must describe actual
containers rather than stale image values from a base Compose file.

Direct exports use a conservative inspect.Size + 5% + 64 MiB ceiling, require
that plus a retained 4 GiB reserve, and check each streamed write and postcondition.
Do not equate free>=4 GiB with sufficient capacity. Also budget a fresh snapshot,
temporary validation copy and private rollback material. Snapshot estimates are
not write bounds. Recheck actual free bytes after any cleanup before proceeding.

## Exact-byte quarantine receipt

A failed stage is not a fresh COMPLETE database backup. Before deleting one,
preserve every opaque file and uid/gid/mode/mtime/xattrs. Avoid parsing/logging
env values or user records. Full file SHA and before/after source CAS must bind
the copy. A deduplicated bundle may use hash-pinned local archive members and
blocks, but pins must survive old-path rotation. Keep receipt, layout, blob pack,
indexes and pinned bases together with private filesystem permissions.

The generic, relocatable reader is tracked here:

```sh
python3 -B ops_failed_stage_quarantine.py verify \
  --bundle "$private_bundle" --proof-sha256 "$reviewed_proof_sha256"
```

It verifies receipt/COMPLETE hashes, all assets and hash-named pinned bases,
then reconstructs raw file hashes from opaque segments without extracting user
data or secrets. Historic absolute base paths in metadata are not followed.
No SQL restore or runtime start is implied by success.

## Explicit bounded cleanup

`ops_failed_stage_cleanup.py` requires explicit source parent, every attempt
name, proof hash, expected active operation/attempt and minimum retained COMPLETE
count. The private request includes the raw pinned proof and compressed layout;
metadata/file hashes must match those receipts. It opens the existing backup
lock without creating/chmodding one, refuses active service, symlinks, unrecorded
entries, changed SHA/identity/metadata or a completion marker at a target root.
Only a plan runs without `--apply-proof-sha`:

```sh
python3 -B ops_failed_stage_cleanup.py \
  --backup-root "$backup_root" --source-parent "$reviewed_source_parent" \
  --attempt "$reviewed_attempt_a" --attempt "$reviewed_attempt_b" \
  --proof-sha256 "$reviewed_proof_sha256" --operation-id "$expected_operation_id" \
  --active-attempt "$protected_active_attempt" --minimum-complete "$retained_complete_count" \
  < "$private_cleanup_request"
```

List **all** authorized attempt names; the two shown are syntax examples, not
permission or a production target. Inspect plan paths/free/allocated bytes.
After separate exact-scope approval, append `--apply-proof-sha` with the same
reviewed hash. The tool rechecks under the lock and checks exact membership and file/directory identity again, unlinks only recorded verified files, and removes directories bottom-up with rmdir. A new unrecorded entry is never recursively removed.
It retains the active attempt, operation record and all recorded COMPLETE sets.
No global staging cleanup or Docker prune belongs in this operation.

## Rollback and later verification

Retain previous source/config/index and all runtime archives. Before reverting
a config pointer, hold the same lock with service inactive and check expected
source/config SHA. Restore only those reviewed bytes; no API restart is needed.

For failed-stage rollback, connect `ops_failed_stage_quarantine.py emit` to a
strict `receive` invocation using **binary subprocess pipes**, never Windows
PowerShell text redirection. Receiver requires proof hash, exact target parent,
backup root and inactive service. It refuses every existing target tree, locks
backup, checks source scope/owners and byte hashes, preserves metadata and
retains 4 GiB before/during writes. It writes failed staging trees only. Restored
inode/ctime cannot equal the deleted original and are not promised. This is not
a production database restore. Budget bytes and transfer time before starting.

After a separately authorized source/config deployment, run read-only preflight
again. An actual systemd backup and isolated SQL/files restore must be recorded
separately from tests, plan output and text checks. If interactive sudo is the
only missing step, request exactly `sudo systemctl start --no-block
arborscan-backup.service`; do not request passwords in chat. Verify exit status,
COMPLETE/SHA and current runtime closure before Windows pull or retention. An
independent always-on offhost receiver and new-host drill remain separate
resource requirements; a directory on the same Windows PC is not one.
