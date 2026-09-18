# Model quality: segmentation and independent taxon labels

## Scope and current evidence (2026-09-18)

`codex/model-quality` is based on the verified server-history main (`60329e1`).
Main was fast-forwarded and pushed before this work. This branch must not be
merged into main as part of rollout.

Production audited before changes: API v4 `history-d9a2a9f`, image ID
`sha256:29d686e0fb365c5fdd00a22c8f4458010e8efe0bd271bf249bd0844e326cf3ad`.
VPS checkout `/opt/arborscan` remains `5d8176a22fddd6634c266a1ed09f1845019065c5`.
The active CPU YOLO segmentation model has one class `{0: tree}`, weights SHA256
`626f62b97b05fd65275dc3f909c3aec090f6c9957c4ef9f12c4e743a1f75a2bb`.
V4 uses oriented RGB→BGR, YOLO retina masks, nearest-neighbour restoration and
selection of a single tree instance. It does not use rembg or a bbox as a mask.
There are no independent trunk/crown training labels.

Audit: 6 CPUs, 11925 MiB RAM (~9924 available), ~128 GiB free disk, no GPU found;
no previous training worker container. Five indexed contours: four submitted,
one rejected, **zero accepted**. No suitable real training dataset was found.
No real training, improved candidate, production activation, or AR field precision
is claimed. Mock worker tests are software tests, not training evidence.

## Fixed mechanisms

The old worker selected `verified` folders and had a non-atomic request/lock flow.
Its training entry is now disabled; diagnostics remain available. Legacy exporter
is restricted to explicitly marked smoke fixtures. The new API uses immutable
accepted revisions and independent administrator taxon labels. Legacy Flutter
training/model buttons open the v4 registry instead of issuing v3 mutations.

Additive migration `003_model_quality.sql` provides private tables, RLS and
service-only functions. Every HTTP route requires the existing server-side admin
check; write RPCs recheck the actor's role. Taxon labels have immutable parent
history, and never accept/reject a contour. Job/label/snapshot retries check their
operation identity and payload; conflicting edits are rejected. Only one queued,
running or cancellation-pending job is allowed. Worker claims have renewable
90-second leases. Lost leases cannot publish a candidate; expired work is marked
failed when the worker reconnects. Interrupted work is not silently restarted.

Admin app: Profile → «Модели, данные и породы». It shows eligibility/exclusions,
separate taxon confirmation, snapshots, asynchronous jobs, cancellation, metrics,
diagnostic overlays, activation and rollback. Status refreshes every five seconds
without re-exporting data. Account changes invalidate in-flight responses.
Operation UUIDs are kept per account on the device; durable labels, snapshots,
jobs and decisions live on the server. Reinstallation loses only local pending
request UUIDs, not registered server jobs. Cross-device status is server-backed.

## Dataset and evaluation contract

- Only `accepted` revision metadata with an explicit actor/time decision is
  eligible for segmentation. Merely being in `verified` does not admit anything.
- Classification requires its own confirmed species-level taxon and authority/ID.
  A genus label is retained but is not used as a species. Class identity is
  authority plus taxon ID; names and evidence remain in the frozen taxonomy.
- Original bytes, oriented PNG, exact mask, revision, decision, label revision,
  hashes, exporter version, seed, exclusions and split assignment are frozen in
  a checksum-addressed private archive. Later edits cannot change a job's data.
- Exact bytes/pixels and known tree/series IDs are unioned before splitting.
  One revision per identical image is selected. Near-duplicate dHash flags are
  advisory; cross-split flags block activation until a corrected dataset is made.
  Similarity detection is not proof that all unknown series have been found.
- Export keeps all exterior components using zero-width polygon bridges. Holes
  cannot be encoded exactly by this YOLO polygon format. Added/lost pixels and
  whole-mask IoU are measured; <0.995 is excluded. Originals are never modified.
  Thin branches, edges, disconnected components and tiny regions have tests.
- Worker additionally checks the installed Ultralytics float32 decoder,
  resampling and polygon rasterizer. It records full-size and training-resolution
  representation loss; <0.995 full-size or <0.95 resized IoU stops that job.
  Training mask ratio is 1; geometric augmentations are disabled. This is a
  representation bound, not measurement accuracy or a promise that downsampling
  preserves every branch. Diagnostic previews alone are resized to 1280 pixels.
- Selection is limited to the latest 500 indexed revisions and 128 MiB retained
  sample bytes. Explicit owner/revision selection is supported by the snapshot
  API. Missing/out-of-window selections fail rather than silently changing scope.
- Train/validation/test are grouped and seeded. Every classification class must
  occur in each partition and at least two classes are required. Small or
  unbalanced data can prevent a valid split; no leakage is used to make it pass.

One CPU worker launches a separate subprocess: default 1 epoch, 320 px, batch 1,
zero loader workers, maximum 5 epochs/640 px/30 minutes, Docker 2 CPUs and 3 GiB
RAM, no swap. It cannot accept shell commands or client paths. Training pins the
selected baseline ID/hash at enqueue. Baseline and candidate use the same v4
inference preprocessing/postprocessing for evaluation. Test is not used for
best-epoch selection or the activation performance gate.

Segmentation reports per-case IoU/Dice/precision/recall, mask complexity, time,
overlays and subprocess peak RSS (not GPU memory). Operational activation gate:
30 independent train groups, 10 validation, 10 test, no unresolved cross-split
near duplicates and non-regressing validation mean IoU. This minimum gate is not
statistical proof of improvement; a human must inspect cases and uncertainty.
Training completion never changes the shared active pointer. Activation validates
checksum, class schema, loading and smoke inference; SQL generation comparison
prevents lost switches. Each request captures one runtime; new reports carry its
model ID and weight checksum. Older reports are not recalculated. Registry outage
fails model selection explicitly rather than letting processes silently diverge.

Local classification is a separate scratch YOLO classifier with a closed class
list. Reports include macro-F1, per-class support/precision/recall and confusion
matrix. Unknown-species behaviour is **not validated without independent unknown
examples**. It remains experimental and cannot replace Pl@ntNet in this release.

## Pl@ntNet

Official sources: [identify API](https://my.plantnet.org/doc/api/identify),
[taxonomy](https://my.plantnet.org/doc/api/taxonomy),
[changelog](https://my.plantnet.org/doc/references/changelog).
This is an external ranked identification service, not an available finetuning
endpoint. Its score is not measured accuracy of ArborScan. Returned scientific
name, taxon ID when provided, engine version, timestamp and original prediction
are kept separately from administrator confirmation. Query URLs are not retained.
Network errors/low scores do not invent a species; genus/author text does not
become a binomial automatically. A reliable Russian-name mapping is not assumed.
For an already saved contour, original prediction can be recovered only from an
explicitly saved matching report; older absent predictions cannot be reconstructed.

## Checks so far

- Local Python: 74 passed, 1 ultralytics-dependent module skipped, 3 subtests;
  two pre-existing Starlette/AnyIO deprecation warnings.
- PGlite executed migration 003 and rerun, ACL, independent labels, conflicting
  retries, cancellation, expired leases, immutable completion, activation CAS and
  rollback using synthetic fixtures. This is **not** a production migration.
- Flutter: 42 tests passed. Analyzer: 106 existing issues (0 errors, 7 warnings,
  99 info); no new issues. Debug APK built without dart-define.
- Production migration, candidate-container tests, rollout, new APK installation
  and phone acceptance are pending unless a later dated entry records completion.

### Additional checks on 2026-09-18

Candidate `8fd7c28` built in `/home/arborscan/model-quality-8fd7c28` on the unchanged
history runtime. Its isolated container passed 76 tests and 3 subtests, without
skips. Real copied production weights passed loading/smoke inference, isolated
runtime selection/rollback and corruption rejection. The real production pointer
and original weight checksum were unchanged. Scratch `yolo11n-cls.yaml` architecture
was instantiated successfully; no training was run.

Actual installed Ultralytics resampler/rasterizer checks on synthetic shapes:
edge rectangle IoU 1.0; disconnected components ~0.99992; a hole ~0.96436 (rejected);
thin branches IoU 1.0 at original resolution but ~0.80825 at 320 px (job rejected).
These are representation tests, not real-tree model quality scores. The app now
offers 320/640 px for the one-epoch trial. Worker reports fidelity errors explicitly
and checks at least 5 GiB free disk before starting. External experiment tracking
is disabled. These follow-up fixes are included in a subsequent commit.

Staging candidate health is 200/ok; unauthenticated new ML/history/contour routes
return 401. Both existing public HTTPS health endpoints remain 200/ok. Production
RPC `model_quality_version` still returned 404 at the last check; migration 003
and production rollout have not been claimed as completed.

## Backup and prepared rollout

Verified backup `/home/arborscan/model-quality-backup-20260918`: Docker image tar
(archive reread), private container inspection, env, four compose files, Git state,
baseline checksum, `data-before.json`, checked `SHA256SUMS`. New ML tables and
storage prefix were confirmed absent before migration. This is not a backup of
all existing user data; this rollout does not modify original contours or reports.
Migration is additive and need not be removed when rolling application code back.

User SQL step: execute `003_model_quality.sql` completely, then
`SELECT public.model_quality_version() AS version;` (expected 1).
After confirmation, independently verify RPC, build an exact commit in a separate
worktree using the existing recorded image as `BASE_IMAGE`, run container tests
and the staging smoke before replacing API v4. The Dockerfile-specific context
includes only application Python/config, not env files, APKs or caches.

`docker-compose.model-quality.yml` overlays the four recorded compose files,
with `MODEL_QUALITY_IMAGE` set to the tested commit tag. A new directory
`/home/arborscan/model-quality` belongs to UID 1000: worker mounts it read/write,
API read-only. Original `/opt/arborscan/models` remains read-only. Only `api-v4`
and `quality-worker` are started/recreated. V3, HTTPS, models and real decisions
are untouched. `smoke_model_quality.py` uses newly registered fixture accounts,
specific revisions and explicit snapshot selection, cleans its own objects,
and neither trains nor activates.

Prepared rollback (execute on VPS; old runtime/code and all ML data are retained):

```bash
set -eu
b=/home/arborscan/model-quality-backup-20260918
(cd "$b" && sha256sum -c SHA256SUMS)
docker stop arborscan-quality-worker
docker image inspect "$(cat "$b/image-id.txt")" >/dev/null || docker image load -i "$b/runtime.tar"
docker compose -p arborscan-v4 \
 -f /home/arborscan/contour-5af8dd1/deploy-vps/docker-compose.v4.yml \
 -f /home/arborscan/contour-backup-20260916/candidate.yml \
 -f /home/arborscan/measurement-backup-20260916-review/candidate.yml \
 -f /home/arborscan/server-history-backup-20260917/candidate.yml \
 up -d --no-build --no-deps api-v4
curl --fail https://31.57.170.88/api/v4/health
```

APK rollback: build `d9a2a9f` in a separate checkout using the same debug key and
`adb -s R5CY40HNVCP install --no-streaming -r <app-debug.apk>`. Do not uninstall or
clear data. Old client lacks the new admin screens; server ML records remain.

## Phone acceptance (not performed yet)

1. Analyze a photo; edit/save a contour; open the previous report and old PNG.
2. Admin: independently correct the taxon with source/ID and explicit confirmation.
   Check that this does not accept the contour. Accept the specific contour via
   existing moderation; check segmentation and classification eligibility separately.
3. Create a snapshot and inspect exclusions/splits. Until there are independent
   train/validation/test examples, training must stay unavailable. Do not manufacture
   approvals or treat smoke fixtures as real quality data to enable the button.
4. Once sufficient genuine reviewed data exists: run the bounded trial, leave the
   screen, restart the app, inspect restored job status, metrics and overlays.
5. Do not activate production as part of this smoke. Candidate acceptance is a
   separate informed decision; real improvement and unknown-species coverage need
   appropriate datasets and independent evaluation.

## Final prepared version / pending manual SQL (2026-09-18)

- Exact candidate and APK source: `f35d12c946ae78b32de9fc2524ad4124f509c4c4`.
  Candidate image `arborscan-api-v4:quality-f35d12c`, ID
  `sha256:b8e0f8d771d426534ece258ea4449e2c293345815ecca4a0b28a8fbda0dd2844`.
  Separate checkout `/home/arborscan/model-quality-f35d12c`; staging container
  `arborscan-quality-candidate` on localhost:18001 is running/healthy.
- Follow-up worker changes passed 5 targeted tests in the real container runtime;
  the preceding full container run passed 76 tests. Local updated suite passed
  74 tests + 3 subtests, with one ultralytics-dependent module skipped.
- Flutter full run: 42 passed before the final dialog guard; all 5 focused ML
  tests passed after it, including hiding an open dialog on account change.
  Analyzer still 106 existing issues. Final debug APK build succeeded.
- Final APK installed on S24 Ultra `R5CY40HNVCP` with `install --no-streaming -r`
  → **Success**, after the user confirmed USB debugging. No uninstall/data clear.
  SHA256 `4833e1ce6af6f8d909682cf1b14ffe53d732e727221a68638ca53b6f05365ecb`.
  Installation is not manual acceptance of the new feature.
- Production remains **history-d9a2a9f**, its original image ID recorded above.
  Both public HTTPS health endpoints still return ok. No worker is deployed,
  no training has run, no active model was changed. SQL RPC 003 last returned 404.
  Authenticated ML integration smoke is prepared, **not yet executed**, because
  the user SQL step is pending. This blocks production replacement, not local work.

After the user executes migration 003 and reports version 1, these are the
prepared VPS commands. They have **not** been executed as a rollout:

```bash
set -eu
b=/home/arborscan/model-quality-backup-20260918
c=/home/arborscan/model-quality-f35d12c
(cd "$b" && sha256sum -c SHA256SUMS)
test "$(docker inspect -f '{{.Image}}' arborscan-api-v4)" = "$(cat "$b/image-id.txt")"
test "$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' arborscan-api-v4:quality-f35d12c)" = f35d12c946ae78b32de9fc2524ad4124f509c4c4
docker exec arborscan-quality-candidate python -c "from arborscan_v4.model_quality_api import QualityStore; assert QualityStore().request('POST','rpc/model_quality_version',json={})==1; print('Migration 003 ready')"
docker exec -i -e MODEL_QUALITY_SMOKE_BASE=http://127.0.0.1:8001 arborscan-quality-candidate python - < "$c/deploy-vps/smoke_model_quality.py"
export MODEL_QUALITY_IMAGE=arborscan-api-v4:quality-f35d12c
docker compose -p arborscan-v4 \
 -f /home/arborscan/contour-5af8dd1/deploy-vps/docker-compose.v4.yml \
 -f /home/arborscan/contour-backup-20260916/candidate.yml \
 -f /home/arborscan/measurement-backup-20260916-review/candidate.yml \
 -f /home/arborscan/server-history-backup-20260917/candidate.yml \
 -f "$c/deploy-vps/docker-compose.model-quality.yml" \
 up -d --no-build --no-deps api-v4 quality-worker
```

After startup, inspect container health and filtered startup errors, both public
HTTPS health endpoints, 401 on unauthenticated ML routes, and run the fixture-only
smoke through public HTTPS:

```bash
docker inspect -f '{{.State.Health.Status}}' arborscan-api-v4 arborscan-quality-worker
curl --fail https://31.57.170.88/api/v3/health
curl --fail https://31.57.170.88/api/v4/health
curl -s -o /dev/null -w '%{http_code}\n' https://31.57.170.88/api/v4/v4/model-quality/status
docker exec -i arborscan-api-v4 python - < /home/arborscan/model-quality-f35d12c/deploy-vps/smoke_model_quality.py
```

If staging verification fails, do not replace production. If production checks
fail, use the rollback above. Leave the additive tables and private artifacts in
place. There is no production candidate ready for activation: suitable accepted
real-tree data and independent evaluation remain missing.
