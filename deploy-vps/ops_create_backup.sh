#!/bin/sh
# Snapshot stage only. ops_backup_policy.py owns locking/publication/rotation.
set -eu
umask 077
target=${1:?Private stage directory required}
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# API image is not replaced; its runtime supplies the installed dependencies.
timeout 1800 docker exec -i arborscan-api-v4 python - < "$script_dir/ops_snapshot.py" > "$target/application.partial.tar"
mv "$target/application.partial.tar" "$target/application.tar"
python3 "$script_dir/ops_verify_restore.py" "$target/application.tar" "$target/restored" > "$target/restore-result.json"
docker inspect arborscan-api arborscan-api-v4 arborscan-quality-worker > "$target/containers.private.json"
cp /etc/arborscan/arborscan.env "$target/arborscan.env.private"
chmod 600 "$target/arborscan.env.private"
git -C /opt/arborscan rev-parse HEAD > "$target/vps-git.txt"
git -C /opt/arborscan status --short > "$target/vps-status.txt"
# Runtime environment resides in private inspection; no env is loaded on restore.
tar -cf "$target/local-files.tar" -C / opt/arborscan/models home/arborscan/model-quality
tar -tf "$target/local-files.tar" > "$target/local-files-list.private.txt"
python3 - "$target" <<'PY'
import json,pathlib,shutil,sys
root=pathlib.Path(sys.argv[1]); configs=root/'compose'; configs.mkdir(mode=0o700)
copied=set()
for c in json.loads((root/'containers.private.json').read_text()):
    paths=c['Config']['Labels'].get('com.docker.compose.project.config_files','').split(',')
    for p in paths:
        if p and p not in copied:
            if not pathlib.Path(p).is_file(): raise SystemExit('Compose source missing; snapshot incomplete')
            dest=configs/(str(len(list(configs.iterdir())))+'-'+pathlib.Path(p).name)
            shutil.copyfile(p,dest);dest.chmod(0o600);copied.add(p)
PY
# Native transaction-consistent database dump, including schema/ACL and roles.
# Fail the overall backup if this stage fails; never publish an incomplete COMPLETE.
python3 "$script_dir/ops_pg_backup.py" "$target/postgres"
echo 'Snapshot stages succeeded; awaiting runtime/manifest verification and publication'
