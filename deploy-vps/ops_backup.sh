#!/bin/sh
# Existing systemd entry point; policy owns the same private backup.lock.
set -eu
umask 077
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$script_dir/ops_backup_policy.py" run \
  --root "${ARBORSCAN_BACKUP_DIR:-/home/arborscan/ops-backups}" "$@"
