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

The policy from `75135519f44d88b286a475499f4b02bb3646c9a3` was installed at
18:07 UTC using the existing lock and an inactive service. Only the policy
changed: SHA `c5c82730b384b090a33f4b6ad6a6e68b6faf44c3e05ad29a9b1dbc90b4ade3e2`
to `a7294e65dd69cc338409c70032d023f878db27b85f44fb9dd93cbbe3e9bd7b0a`.
Other tool/config hashes and production container identities stayed unchanged.
Direct replay through the same dispatcher completed at 18:19:45 UTC: exit0,
`verified_existing`, same snapshot/manifest, `deleted:[]`, stderr0 bytes,
692.7 seconds. The previous systemd run took 2222 seconds. These are different
invocations/environments, not a controlled performance benchmark. The new run
was direct CLI; do not describe it as another systemd execution. The next
scheduled systemd invocation of the newly installed source is not yet observed.

Prepared rollback, **not executed**:

```sh
python3 -B /home/arborscan/ops-tools-policy-rollback-20261117-75135519/rollback-policy.py
```

Its SHA is `45d20f90334e347ad1bfed50b4f5e24c7f95e8b89b22fb6339099a1d9a3bba40`.
It requires the existing backup lock, inactive service, current new-policy SHA
and verified old bytes before atomic replacement. It refuses concurrent backup
or source drift; no force, configuration/index change or application restart.
The preceding runtime/index rollback and exact-seven quarantine remain separate.

Both selected sets `20261115T032812Z` and `20261117T033225Z` were copied to the
existing `D:/ArborScanBackups` and fully verified with native/runtime checks by
the unchanged installed Windows reader. Its retention policy then removed only
the reviewed `20261022T033459Z` and `20261025T033213Z`. Two verified COMPLETE sets,
eight shared runtime archives and the quarantine pins remain. Read-only HTTPS
health returned 200/status:ok for v3/v4 with normal certificate verification.
Safe proof: [checks.json](evidence/remediation-20261117/checks.json). This does
not create an independent always-on receiver or prove recovery on a new host.

## Scheduled Windows reader: select before expensive inventory

The installed reader is newer than the repository's historical reader. Keep it:
`ops_local_retention.adapt_reader_source` changes only its `main`/`read_inventory`
hooks, preserving other function ASTs and the original inventory program.
With the existing explicit latest-two policy, load policy first and choose two
eligible snapshots from bounded manifest/native/runtime metadata before building
block maps or hashing runtime assets. All-container live/protection discovery
still runs first. A malformed modern newest COMPLETE refuses; it is not hidden
by falling back to an older snapshot. Clearly legacy sets remain distinguishable.
Without the policy the original remote program/transport bytes are unchanged.

This is selection, not a cached integrity receipt: selected full SHA, native,
runtime and destination/retention verification remain mandatory. Metadata reads
are bounded to 4 MiB outer manifest, 1 MiB native manifest, 2 MiB runtime/marker.
The 28-set fixture reproduced 56 unnecessary heavy reads; the fix reads four
selected payload/runtime files. The scoped Windows compatibility suite passed
75 tests. Installation must preserve exact old bytes under the existing local
pull lock and compare all other installed tool hashes; do not overwrite the
newer reader with the older repository copy. A standard CLI run with the
default 180-second inventory timeout is separate evidence from these tests.

Actual CAS install of `aaf6d65b` completed on 17 November at 19:13 UTC. Only
`ops_local_retention.py` (SHA `1ba3a1fc…662ce`) and the adapted newer reader
(SHA `65a4501b…936ee`) changed. Exact old bytes/metadata/ACL and CAS rollback
were retained privately under the existing backup root. Other tools/config/
policy and scheduled-task XML hashes remained unchanged. The standard installed
CLI, with `--retention-dry-run` and default inventory timeout180, completed
exit0 in102.061s: both sets verified_existing, zero deletion candidates, no
retry/monkeypatch/timeout override. This was the actual entry point, not a new
calendar-triggered task run. [Proof](evidence/remediation-20261117/windows-scheduled.json).

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

Native Docker-save framing can contain directory headers and use different raw
headers/EOF padding from Python's canonical tar writer. Do not claim the native
archive SHA after rewriting it with the original file-only format. The explicit
format-2 reader preserves each checked 512-byte header, order and bounded zero
EOF without filesystem extraction:

```sh
python3 -B ops_image_delta.py apply-raw-member "$base_archive" "$native_delta" "$new_partial_archive" \
  --image-id "$actual_image_id" --base-sha256 "$base_sha256" \
  --expected-archive-sha256 "$native_archive_sha256"
```

The trusted full target SHA is mandatory here. Only REG/AREG files and zero-size
directories are accepted; links, sparse/contiguous types, unsafe paths, duplicate
or file/ancestor collisions, bad checksums/sizes/EOF refuse. The output budget
and 4 GiB floor are checked before creation and per write. The existing image
config/platform/blob and full archive checks remain mandatory. The default
`apply-member` remains format 1 and refuses format 2 before creating an output.
Software evidence: 55 focused tests (37 native-format cases and 18 previous
image/member cases), tiny synthetic OCI data with three public Docker headers.
This unit proof does not itself transfer a production archive or load Docker.

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

Actual 17 November continuation: a new pre-deployment capture (not a daily
replay or systemd invocation) published `20261117T200418Z`, manifest SHA
`52a447b85e5849ebc22998d3cd4bfe4a2e48619952ea4f4a64dbc225167c4214`,
exit 0 in 1090.420 seconds. PostgreSQL/JSON share the exported snapshot and
the seven current runtime identities remain unchanged. Existing VPS retention14
deleted only `20261014T032748Z`; all other retained manifests stayed unchanged.
The standard installed Windows pull succeeded in 178.898 seconds with the
default 180-second inventory limit. Existing latest-two apply removed only
`20261115T032812Z` after full verification; final retained sets are
`20261117T033225Z` and `20261117T200418Z`, with eight shared runtime archives.
Installed tools, policy and scheduled-task definition stayed unchanged. This
is not a fresh SQL restore: the actual isolated restore used the morning set.
[Final proof](evidence/remediation-20261117/fresh-backup.json).

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
