"""Push verified native/Storage/runtime sets to an authorized private SSH host.

Configuration, SSH key and known_hosts stay outside Git. Receipts are written
only after reading every byte back over SSH and hashing it locally. A receipt
from a previous run is insufficient for pre-deletion: use `verify` immediately.
The existing Windows pull job is independent and is not changed by this tool.
"""
import argparse
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import socket
import stat
import subprocess
import tempfile
import time

from ops_offsite_receiver import CHUNK, NAME, encoded, private_path, sha, validate
from ops_verify_release_bundle import verify as verify_release
from ops_verify_restore import restore as restore_application


def manifest_files(root):
    result = {}
    for row in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = row.split("  ", 1)
        name = str(PurePosixPath(name))
        if not re.fullmatch(r"[a-f0-9]{64}", expected) or name in result:
            raise ValueError("invalid_manifest")
        path = private_path(root, name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("source_content_not_verified")
        result[name] = path
    return result


def runtime_images(backup):
    inventory = json.loads((backup / "containers.private.json").read_bytes())
    names = {"/arborscan-api-v4", "/arborscan-quality-worker"}
    selected = {r["Name"]: r["Image"] for r in inventory if r["Name"] in names}
    if selected.keys() != names or not all(re.fullmatch(r"sha256:[a-f0-9]{64}", image) for image in selected.values()):
        raise ValueError("runtime_image_inventory_missing")
    return set(selected.values())


def build_package(backup, release_root):
    backup = Path(backup).absolute()
    if not NAME.fullmatch(backup.name) or not private_path(backup, "COMPLETE").is_file():
        raise ValueError("incomplete_source_backup")
    files = {backup.name + "/" + name: path for name, path in manifest_files(backup).items()}
    for name in ("SHA256SUMS", "COMPLETE", "postgres/SHA256SUMS"):
        path = private_path(backup, name)
        if path.is_file():
            files[backup.name + "/" + name] = path
    images = runtime_images(backup)
    runtime_assets = []
    runtime_path = private_path(backup, "RUNTIME_DEPENDENCIES.json")
    if runtime_path.is_file():
        if backup.name + "/RUNTIME_DEPENDENCIES.json" not in files:
            raise ValueError("runtime_dependencies_not_hash_covered")
        runtime = json.loads(runtime_path.read_bytes())
        if runtime.get("format") != 1 or set(runtime.get("images", {})) != images:
            raise ValueError("runtime_index_image_mismatch")
        for assets in runtime["images"].values():
            if not isinstance(assets, list) or not assets:
                raise ValueError("runtime_archive_missing")
            for asset in assets:
                path = Path(asset["path"])
                expected = asset["sha256"]
                if not path.is_absolute() or not re.fullmatch(r"[a-f0-9]{64}", expected):
                    raise ValueError("unsafe_runtime_archive")
                path = private_path(path.parent, path.name)
                if not path.is_file() or sha(path) != expected:
                    raise ValueError("runtime_archive_checksum_mismatch")
                name = "runtime-assets/" + expected + ".archive"
                files[name] = path
                item = {"source_path": str(path), "package_path": name, "sha256": expected}
                if item not in runtime_assets:
                    runtime_assets.append(item)
    else:
        # Historical release-index format is preserved and must also read every
        # actual image/source/base/overlay archive. Missing current mapping fails.
        release_root = Path(release_root).absolute()
        verify_release(backup, release_root)
        index_path = private_path(release_root, "RELEASE_INDEX.json")
        index = json.loads(index_path.read_bytes())
        files["RELEASE_INDEX.json"] = index_path
        for image in images:
            release = private_path(release_root, index["images"][image]["release"]).parent
            for name, path in manifest_files(release).items():
                files[path.relative_to(release_root).as_posix()] = path
            for name in ("SHA256SUMS", "COMPLETE"):
                path = private_path(release, name)
                files[path.relative_to(release_root).as_posix()] = path
    package = {"format": 1, "name": backup.name,
               "manifest_sha256": sha(backup / "SHA256SUMS"),
               "files": {name: {"sha256": sha(path), "bytes": path.stat().st_size}
                         for name, path in sorted(files.items())}}
    if runtime_assets:
        package["runtime_assets"] = runtime_assets
    validate(package)
    return package, files


def verify_relocated_runtime(package, destination):
    """Preserve hash-covered absolute source paths; verify a separate relocation map."""
    backup = destination / package["name"]
    runtime = json.loads((backup / "RUNTIME_DEPENDENCIES.json").read_bytes())
    images = runtime_images(backup)
    if runtime.get("format") != 1 or set(runtime.get("images", {})) != images:
        raise ValueError("runtime_index_image_mismatch")
    mapping = {(item["source_path"], item["sha256"]): item["package_path"]
               for item in package.get("runtime_assets", [])}
    relocated = []
    for assets in runtime["images"].values():
        if not isinstance(assets, list) or not assets:
            raise ValueError("runtime_archive_missing")
        for asset in assets:
            key = (asset["path"], asset["sha256"])
            if key not in mapping:
                raise ValueError("runtime_relocation_missing")
            path = private_path(destination, mapping[key])
            if sha(path) != asset["sha256"]:
                raise ValueError("runtime_archive_checksum_mismatch")
            relocated.append({"original_path": asset["path"], "restored_path": str(path),
                              "sha256": asset["sha256"]})
    (destination / "RELOCATED_RUNTIME.json").write_text(json.dumps({"format": 1, "assets": relocated}))
    return len(relocated)


def load_config(path):
    path = Path(path).absolute()
    private_path(path.parent, path.name)
    if os.name != "nt" and (stat.S_IMODE(path.stat().st_mode) & 0o077):
        raise ValueError("private_config_permissions_required")
    config = json.loads(path.read_bytes())
    required = {"format", "destination_id", "host", "user", "port", "remote_root",
                "receiver_script", "identity_file", "known_hosts_file",
                "is_independent", "always_on"}
    if set(config) != required or config["format"] != 1:
        raise ValueError("invalid_private_config")
    if config["is_independent"] is not True or config["always_on"] is not True:
        raise ValueError("independent_always_on_destination_required")
    if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,64}", config["destination_id"]):
        raise ValueError("invalid_destination_id")
    if (not re.fullmatch(r"[a-zA-Z0-9_.:-]+", config["host"])
            or not re.fullmatch(r"[a-z_][a-z0-9_-]*", config["user"])
            or type(config["port"]) is not int or not 1 <= config["port"] <= 65535):
        raise ValueError("invalid_private_config")
    addresses = {r[4][0] for r in socket.getaddrinfo(config["host"], config["port"], type=socket.SOCK_STREAM)}
    if not addresses or addresses & {"31.57.170.88", "127.0.0.1", "::1", "0.0.0.0"}:
        raise ValueError("destination_not_independent")
    for key in ("remote_root", "receiver_script"):
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", config[key]) or ".." in Path(config[key]).parts:
            raise ValueError("invalid_remote_path")
    for key in ("identity_file", "known_hosts_file"):
        local = Path(config[key])
        if not local.is_absolute():
            raise ValueError("private_path_required")
        private_path(local.parent, local.name)
        if not local.is_file() or (os.name != "nt" and stat.S_IMODE(local.stat().st_mode) & 0o077):
            raise ValueError("private_credentials_permissions_required")
    return config


class SSHTransport:
    def __init__(self, config):
        self.config = config
        # A full immutable runtime can be many GiB. Reuse one authenticated
        # connection instead of performing thousands of key exchanges per chunk.
        self.control_directory = tempfile.TemporaryDirectory(prefix="arborscan-offsite-ssh-") if os.name == "posix" else None
        self.control_path = str(Path(self.control_directory.name) / "control") if self.control_directory else None

    def close(self):
        if self.control_directory is not None:
            c = self.config
            try:
                subprocess.run(["ssh", "-S", self.control_path, "-O", "exit", c["user"] + "@" + c["host"]],
                               stdin=subprocess.DEVNULL, capture_output=True, timeout=15)
            except (OSError, subprocess.TimeoutExpired):
                pass
            self.control_directory.cleanup()
            self.control_directory = None

    def __call__(self, request, body=b""):
        c = self.config
        command = "python3 " + shlex.quote(c["receiver_script"]) + " --root " + shlex.quote(c["remote_root"])
        args = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=15",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
                "-o", "UserKnownHostsFile=" + c["known_hosts_file"],
                "-i", c["identity_file"], "-p", str(c["port"]), c["user"] + "@" + c["host"], command]
        if self.control_path is not None:
            args[1:1] = ["-o", "ControlMaster=auto", "-o", "ControlPersist=60", "-o", "ControlPath=" + self.control_path]
        raw = json.dumps(request, separators=(",", ":")).encode() + b"\n" + body
        # No key/password/response/archive bytes ever enter a log or exception.
        for attempt in range(3):
            try:
                result = subprocess.run(args, input=raw, capture_output=True, timeout=180)
                if result.returncode:
                    raise ValueError("private_transport_failed")
                reply = json.loads(result.stdout)
                if reply.get("ok") is not True:
                    raise ValueError("private_receiver_failed")
                return reply["result"]
            except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
                # A write can have succeeded despite a lost reply. Do not retry
                # mutating RPCs blindly; a fresh transfer probes the actual size.
                if request["action"] in ("put", "begin", "commit", "reset_partial") or attempt == 2:
                    raise ValueError("private_transport_failed") from None
                time.sleep(1 + attempt)


def readback(package, rpc, destination=None):
    """Hash actual downloaded bytes. Optionally recover into a fresh directory."""
    if destination is not None:
        destination = Path(destination).absolute()
        if destination.exists():
            raise ValueError("recovery_destination_must_be_new")
        private_path(destination, "restore")
        destination.mkdir(mode=0o700, parents=True)
    count = total = 0
    for name, entry in package["files"].items():
        digest = hashlib.sha256()
        target = private_path(destination, name) if destination is not None else None
        if target is not None:
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            stream = target.open("wb")
        else:
            stream = None
        try:
            for offset in range(0, entry["bytes"], CHUNK):
                length = min(CHUNK, entry["bytes"] - offset)
                reply = rpc({"action": "read", "name": package["name"], "sha256": entry["sha256"],
                             "offset": offset, "length": length})
                raw = base64.b64decode(reply["data"], validate=True)
                if len(raw) != length:
                    raise ValueError("short_remote_read")
                digest.update(raw)
                if stream is not None:
                    stream.write(raw)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError("readback_checksum_mismatch")
        finally:
            if stream is not None:
                stream.close()
        count += 1
        total += entry["bytes"]
    if destination is not None:
        # Native dump/Storage/config/runtime use the same immutable package.
        backup = destination / package["name"]
        manifest_files(backup)
        manifest_files(backup / "postgres")
        if (backup / "RUNTIME_DEPENDENCIES.json").is_file():
            verify_relocated_runtime(package, destination)
        else:
            verify_release(backup, destination)
        linked = restore_application(backup / "application.tar", destination / "restored-application")
        (destination / "EXTERNAL_RECOVERY_VERIFIED.json").write_text(json.dumps({
            "files": count, "bytes": total, "application_restore": linked,
            "database_restore_executed": False}))
    return {"files": count, "bytes": total}


def transfer(package, files, rpc):
    validate(package)
    rpc({"action": "begin", "name": package["name"], "package": package})
    seen = set()
    for name, entry in package["files"].items():
        digest = entry["sha256"]
        if digest in seen:
            continue
        seen.add(digest)
        request = {"name": package["name"], "sha256": digest}
        probe = rpc({**request, "action": "probe"})
        if probe["complete"]:
            continue
        offset = probe["bytes"]
        if type(offset) is not int or offset < 0:
            raise ValueError("invalid_partial_size")
        if offset > entry["bytes"]:
            rpc({**request, "action": "reset_partial"})
            offset = 0
            probe["prefix_sha256"] = hashlib.sha256(b"").hexdigest()
        with files[name].open("rb") as stream:
            prefix = hashlib.sha256()
            remaining = offset
            while remaining:
                raw = stream.read(min(CHUNK, remaining))
                if not raw:
                    raise ValueError("invalid_partial_size")
                prefix.update(raw)
                remaining -= len(raw)
            if prefix.hexdigest() != probe["prefix_sha256"]:
                rpc({**request, "action": "reset_partial"})
                offset = 0
                stream.seek(0)
            if entry["bytes"] == 0:
                rpc({**request, "action": "put", "offset": 0,
                     "chunk_sha256": hashlib.sha256(b"").hexdigest()}, b"")
            while raw := stream.read(CHUNK):
                rpc({**request, "action": "put", "offset": offset,
                     "chunk_sha256": hashlib.sha256(raw).hexdigest()}, raw)
                offset += len(raw)
    reply = rpc({"action": "commit", "name": package["name"]})
    if reply["package_sha256"] != hashlib.sha256(encoded(package)).hexdigest():
        raise ValueError("remote_package_mismatch")
    return readback(package, rpc)


def receipt(path, package, result, config):
    from ops_offsite_receiver import atomic
    value = {"format": 1, "name": package["name"], "manifest_sha256": package["manifest_sha256"],
             "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "destination_id": config["destination_id"], "is_independent": True,
             "always_on": True, "actual_content_verified": True, "source_content_verified": True,
             "package_sha256": hashlib.sha256(encoded(package)).hexdigest(),
             "assets_sha256": {name: entry["sha256"] for name, entry in package["files"].items()},
             **result}
    path = Path(path).absolute()
    private_path(path.parent, path.name)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    atomic(path, json.dumps(value, sort_keys=True).encode())
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("push", "verify", "recover"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--receipt-root", type=Path, default=Path("/home/arborscan/ops-backups/offsite-receipts"))
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    rpc = None
    try:
        os.umask(0o077)
        config = load_config(args.config)
        rpc = SSHTransport(config)
        identity = rpc({"action": "identity"})
        machine = Path("/etc/machine-id")
        if machine.is_file() and identity.get("machine_id_sha256") == sha(machine):
            raise ValueError("destination_same_host")
        if args.operation == "push":
            package, files = build_package(args.backup, args.release_root)
            result = transfer(package, files, rpc)
        else:
            if not NAME.fullmatch(args.name or ""):
                raise ValueError("invalid_backup_name")
            package = rpc({"action": "inspect", "name": args.name})["package"]
            validate(package)
            result = readback(package, rpc, destination=args.destination if args.operation == "recover" else None)
        receipt(args.receipt_root / (package["name"] + ".json"), package, result, config)
        print(json.dumps({"status": "actual_external_bytes_verified", "name": package["name"], **result}))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        # Deliberately no raw path, provider response, credentials or user bytes.
        reason = str(exc)
        if not re.fullmatch(r"[a-z_]{1,80}", reason):
            reason = "external_backup_not_confirmed"
        print(json.dumps({"status": "failed", "reason": reason}))
        return 1
    finally:
        if rpc is not None:
            rpc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
