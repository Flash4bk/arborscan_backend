# Evidence for the ArborScan audit started 11.11.2026

Scope: AS-01...AS-16, findings before fixes. See
`deploy-vps/PROJECT_AUDIT_20261111.md` and the existing master plan.

This directory contains safe summaries, hashes, synthetic probes and guest
emulator screenshots. Full local logs, rendered PDF pages and the isolated
Python audit environment remain under `output/audit-20261111/`.
No APK, full PDF, credentials, env files, user records, private diagnostics or
third-party project inventory are published here.

Evidence distinguishes:

- 303 passing current Flutter tests, 140 passing backend cases plus one skip,
  seven measurement/profile tests, 48 backup fixture tests;
- five additional probes that deliberately reproduce existing defects;
- public health/unauthorised reads and read-only production metadata;
- real guest UI of APK34 on a separate read-only Android emulator;
- five synthetic local PDF artifacts, 20 rendered/visually checked pages;
- unsuccessful Android unit setup and OSV network check, without claiming
  those checks passed.

The production runtime differs from the isolated Python test environment.
Mocked storage, SQL text review, synthetic PDFs, guest UI, and prior user
acceptance are separate evidence classes. No production writes, DB migration,
backup/restore execution, model activation/training or S24 installation occurred.

## Reproducing the defect probes

From `arborscan_app`, use the repository's existing Flutter package configuration:

```powershell
flutter test --no-pub ../deploy-vps/evidence/project-audit-20261111/probes/map_selector_probe_test.dart
flutter test --no-pub ../deploy-vps/evidence/project-audit-20261111/probes/legacy_verdict_probe_test.dart
```

These probes assert the observed bad behaviour. Their green result confirms
reproduction, not correctness. Do not add them unchanged to the normal CI suite;
an implementation fix needs regression assertions for the desired behaviour.

`checksums.json` hashes every evidence file except itself. `.gitattributes`
preserves bytes so that checked-in blobs retain those SHA256 values.
