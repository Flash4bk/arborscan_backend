# ArborScan Unified Analysis v4 — backend alpha

This package is intentionally deployed **beside** the existing v3 API.

- v3: `127.0.0.1:8000` — untouched stable baseline
- v4: `127.0.0.1:8001` — accuracy-first unified analysis

## What v4 removes immediately

The new pipeline never creates metric measurements from:

- average species height;
- an assumed 1.7 m person;
- an assumed 1.5 m car;
- an assumed 1 m stick;
- a fake rectangular tree mask when segmentation fails;
- a fallback mechanical profile of another species.

If no physical calibration exists, photo analysis returns pixel geometry and
`absolute_scale_required` instead of invented metres.

## Current v4 scope

The first alpha unifies:

1. tree segmentation from `model_v4.pt`;
2. Pl@ntNet species identification;
3. AR measurements supplied by the app;
4. explicit reference/manual photo scale;
5. height/crown/trunk result fusion with confidence and provenance.

Risk remains deliberately disabled until metric measurements and species
mechanical profiles are validated.

## Files to copy into repo

Copy these paths into `/opt/arborscan`:

```text
arborscan_v4/
deploy-vps/Dockerfile.v4
deploy-vps/Dockerfile.v4.dockerignore
deploy-vps/docker-compose.v4.yml
```

The test directory is for local/repository testing and does not enter the image.

## Server commands

```bash
cd /opt/arborscan

docker compose -f deploy-vps/docker-compose.v4.yml build

docker compose -f deploy-vps/docker-compose.v4.yml up -d

docker compose -f deploy-vps/docker-compose.v4.yml ps

curl -sS http://127.0.0.1:8001/health | python3 -m json.tool
curl -sS 'http://127.0.0.1:8001/health?deep=true' | python3 -m json.tool
```

## Photo-only smoke test

```bash
curl -sS --max-time 180 \
  -X POST http://127.0.0.1:8001/v4/analyze-tree \
  -F 'file=@/tmp/tree_test.jpg;type=image/jpeg' \
  -F 'include_images=false' \
  -o /tmp/v4_result.json \
  -w '\nHTTP_CODE=%{http_code}\n'

python3 -m json.tool /tmp/v4_result.json
```

Expected behaviour without AR/reference scale:

```text
analysis_status = absolute_scale_required
measurements.height.value_m = null
measurements.crown_width.value_m = null
measurements.trunk_diameter.value_m = null
```

## AR-assisted smoke test

Use known test values only to verify fusion mechanics:

```bash
curl -sS --max-time 180 \
  -X POST http://127.0.0.1:8001/v4/analyze-tree \
  -F 'file=@/tmp/tree_test.jpg;type=image/jpeg' \
  -F 'ar_height_m=18.20' \
  -F 'ar_crown_width_m=8.40' \
  -F 'ar_trunk_diameter_m=0.61' \
  -F 'ar_trunk_measurement_height_m=1.30' \
  -F 'ar_quality=0.90' \
  -F 'include_images=false' \
  | python3 -m json.tool
```

This test does **not** validate AR accuracy; it validates that backend does not
replace the supplied values with species statistics and that DBH is labelled
only when the measurement height is approximately 1.3 m.
