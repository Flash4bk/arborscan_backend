#!/bin/sh
# Run as arborscan; private, nonoverlapping snapshots. Never removes old copies.
set -eu
umask 077
base=${ARBORSCAN_BACKUP_DIR:-/home/arborscan/ops-backups}
mkdir -p "$base"
chmod 700 "$base"
exec 9>"$base/backup.lock"
flock -n 9 || exit 0
# Bounded retention without deleting data: stop and require archival after 14 copies.
copies=$(find "$base" -mindepth 2 -maxdepth 2 -name COMPLETE | wc -l)
[ "$copies" -lt 14 ] || { echo 'Backup refused: 14 copies retained; archive elsewhere before continuing' >&2; exit 1; }
available=$(df -Pk "$base" | awk 'NR==2 {print $4}')
[ "$available" -ge 20971520 ] || { echo 'Backup refused: less than 20 GiB free' >&2; exit 1; }
target="$base/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$target"
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# API image is not replaced; its runtime supplies the installed dependencies.
timeout 1800 docker exec -i arborscan-api-v4 python - < "$script_dir/ops_snapshot.py" > "$target/application.partial.tar"
mv "$target/application.partial.tar" "$target/application.tar"
python3 "$script_dir/ops_verify_restore.py" "$target/application.tar" "$target/restored" > "$target/restore-result.json"
docker inspect arborscan-api-v4 arborscan-quality-worker > "$target/containers.private.json"
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
for c in json.loads((root/'containers.private.json').read_text()):
    paths=c['Config']['Labels'].get('com.docker.compose.project.config_files','').split(',')
    for p in paths:
        if p and pathlib.Path(p).is_file():
            dest=configs/(str(len(list(configs.iterdir())))+'-'+pathlib.Path(p).name)
            shutil.copyfile(p,dest);dest.chmod(0o600)
PY
(cd "$target" && find . -type f ! -name SHA256SUMS ! -path './restored/*' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS && sha256sum --quiet -c SHA256SUMS)
touch "$target/COMPLETE"
echo "Completed private application/files snapshot: $target"
