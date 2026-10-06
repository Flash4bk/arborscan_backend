"""Private SSH backup receiver. All bytes are verified; stdout is an RPC result.

Install this on an explicitly authorized independent host, owned by the backup
user. The SSH account and directories must not be public. Nothing here deletes
completed backups or application data. Files are content addressed so immutable
runtime archives need to cross the network only once.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

CHUNK = 4 * 1024 * 1024
NAME = re.compile(r"\d{8}T\d{6}Z")
DIGEST = re.compile(r"[a-f0-9]{64}")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encoded(package):
    return json.dumps(package, sort_keys=True, separators=(",", ":")).encode()


def validate(package):
    if package.get("format") != 1 or not NAME.fullmatch(package.get("name", "")):
        raise ValueError("invalid_package")
    if not DIGEST.fullmatch(package.get("manifest_sha256", "")):
        raise ValueError("invalid_package")
    files = package.get("files")
    if not isinstance(files, dict) or not files or len(files) > 10000:
        raise ValueError("invalid_package")
    for name, entry in files.items():
        relative = PurePosixPath(name)
        if (not re.fullmatch(r"[A-Za-z0-9_./-]+", name) or relative.is_absolute()
                or ".." in relative.parts or str(relative) != name
                or not DIGEST.fullmatch(entry.get("sha256", ""))
                or type(entry.get("bytes")) is not int or entry["bytes"] < 0):
            raise ValueError("invalid_package")
    required = {package["name"] + "/" + name for name in (
        "SHA256SUMS", "COMPLETE", "application.tar", "local-files.tar", "arborscan.env.private",
        "containers.private.json", "postgres/database.dump", "postgres/roles.sql",
        "postgres/source.json", "postgres/archive-list.private.txt", "postgres/SHA256SUMS", "postgres/COMPLETE")}
    if (not required <= files.keys() or ("RELEASE_INDEX.json" not in files
            and package["name"] + "/RUNTIME_DEPENDENCIES.json" not in files)):
        raise ValueError("package_not_full_native_runtime")
    if files[package["name"] + "/SHA256SUMS"]["sha256"] != package["manifest_sha256"]:
        raise ValueError("package_manifest_mismatch")
    if package["name"] + "/RUNTIME_DEPENDENCIES.json" in files:
        assets = package.get("runtime_assets")
        if not isinstance(assets, list) or not assets:
            raise ValueError("runtime_relocation_missing")
        seen_sources = {}
        for asset in assets:
            digest, source = asset.get("sha256", ""), asset.get("source_path", "")
            name = asset.get("package_path", "")
            if (not DIGEST.fullmatch(digest) or not isinstance(source, str) or not source
                    or "\x00" in source or name != "runtime-assets/" + digest + ".archive"
                    or name not in files or files[name]["sha256"] != digest):
                raise ValueError("invalid_runtime_relocation")
            if source in seen_sources and seen_sources[source] != digest:
                raise ValueError("conflicting_runtime_relocation")
            seen_sources[source] = digest


def private_path(root, relative):
    """Reject even symlinks contained in the root; no traversal through them."""
    root = Path(root)
    if not root.is_absolute() or root.is_symlink() or (hasattr(root, "is_junction") and root.is_junction()):
        raise ValueError("unsafe_root")
    for ancestor in (root, *root.parents):
        if ancestor.is_symlink() or (hasattr(ancestor, "is_junction") and ancestor.is_junction()):
            raise ValueError("unsafe_root")
    name = PurePosixPath(relative)
    if name.is_absolute() or ".." in name.parts or "\\" in relative or ":" in relative:
        raise ValueError("unsafe_path")
    target = root / name
    for part in (target, *target.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("unsafe_path")
        if part == root:
            break
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("unsafe_path")
    return target


def atomic(path, data):
    temporary = path.with_name(path.name + ".tmp")
    if temporary.is_symlink():
        raise ValueError("unsafe_path")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def dispatch(root, request, body=b""):
    os.umask(0o077)
    root = Path(root)
    private_path(root, "objects")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt" and (root.stat().st_mode & 0o077 or root.stat().st_uid != os.geteuid()):
        raise ValueError("receiver_root_must_be_private")
    for directory in ("objects", "sets", ".staging"):
        private_path(root, directory).mkdir(exist_ok=True, mode=0o700)
    action = request.get("action")
    if action == "identity":
        machine = Path("/etc/machine-id")
        return {"machine_id_sha256": sha(machine) if machine.is_file() else None}
    name = request.get("name", "")
    if not NAME.fullmatch(name):
        raise ValueError("invalid_backup_name")
    stage = private_path(root, ".staging/" + name)
    final = private_path(root, "sets/" + name)
    private_path(root, "sets/" + name + "/COMPLETE")
    private_path(root, ".staging/" + name + "/COMPLETE")
    if final.exists() and not (final / "COMPLETE").is_file():
        raise ValueError("unowned_final_destination")
    if action == "begin":
        package = request["package"]
        validate(package)
        if package["name"] != name:
            raise ValueError("invalid_package")
        raw = encoded(package)
        current = final if (final / "COMPLETE").is_file() else stage
        current.mkdir(exist_ok=True, mode=0o700)
        package_file = private_path(root, current.relative_to(root).as_posix() + "/package.json")
        if package_file.exists() and package_file.read_bytes() != raw:
            raise ValueError("existing_package_conflict")
        atomic(package_file, raw)
        return {"complete": (final / "COMPLETE").is_file()}
    current = final if (final / "COMPLETE").is_file() else stage
    package_file = private_path(root, current.relative_to(root).as_posix() + "/package.json")
    package = json.loads(package_file.read_bytes())
    validate(package)
    if action in ("commit", "inspect"):
        for entry in package["files"].values():
            target = private_path(root, "objects/" + entry["sha256"])
            if not target.is_file() or target.stat().st_size != entry["bytes"] or sha(target) != entry["sha256"]:
                raise ValueError("remote_content_not_verified")
        if action == "commit" and current == stage:
            atomic(stage / "COMPLETE", b"verified\n")
            os.replace(stage, final)
        elif action == "inspect" and not (final / "COMPLETE").is_file():
            raise ValueError("remote_set_incomplete")
        return {"package": package, "package_sha256": hashlib.sha256(encoded(package)).hexdigest()}
    digest = request.get("sha256", "")
    entries = [e for e in package["files"].values() if e["sha256"] == digest]
    if not DIGEST.fullmatch(digest) or not entries:
        raise ValueError("unlisted_blob")
    if len({e["bytes"] for e in entries}) != 1:
        raise ValueError("conflicting_blob_size")
    total = entries[0]["bytes"]
    target = private_path(root, "objects/" + digest)
    partial = private_path(root, "objects/" + digest + ".partial")
    if action == "probe":
        existing = target if target.is_file() else partial
        if target.is_file() and (target.stat().st_size != total or sha(target) != digest):
            raise ValueError("corrupt_existing_blob")
        return {"complete": target.is_file(), "bytes": existing.stat().st_size if existing.is_file() else 0,
                "prefix_sha256": sha(existing) if existing.is_file() else hashlib.sha256(b"").hexdigest()}
    if action == "reset_partial":
        if target.is_file():
            raise ValueError("completed_blob_immutable")
        partial.unlink(missing_ok=True)
        return {"reset": True}
    offset = request.get("offset")
    if type(offset) is not int or offset < 0:
        raise ValueError("invalid_offset")
    if action == "put":
        if len(body) > CHUNK or offset + len(body) > total or target.exists():
            raise ValueError("invalid_chunk")
        if (partial.stat().st_size if partial.exists() else 0) != offset:
            raise ValueError("chunk_offset_conflict")
        if hashlib.sha256(body).hexdigest() != request.get("chunk_sha256"):
            raise ValueError("chunk_checksum_mismatch")
        with partial.open("ab") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(partial, 0o600)
        if partial.stat().st_size == total:
            if sha(partial) != digest:
                raise ValueError("file_checksum_mismatch")
            os.replace(partial, target)
        return {"bytes": offset + len(body)}
    if action == "read":
        length = request.get("length")
        if type(length) is not int or not 0 <= length <= CHUNK or offset + length > total:
            raise ValueError("invalid_chunk")
        with target.open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(length)
        if len(raw) != length:
            raise ValueError("short_remote_read")
        return {"data": base64.b64encode(raw).decode("ascii")}
    raise ValueError("unsupported_action")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    # A private lock covers each RPC, including simultaneous source processes.
    try:
        import fcntl
        os.umask(0o077)
        private_path(args.root, "receiver.lock")
        args.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (args.root / "receiver.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            line = sys.stdin.buffer.readline(1024 * 1024 + 1)
            if len(line) > 1024 * 1024:
                raise ValueError("request_too_large")
            request = json.loads(line)
            body = sys.stdin.buffer.read(CHUNK + 1)
            result = dispatch(args.root, request, body)
            print(json.dumps({"ok": True, "result": result}))
    except (ValueError, OSError, KeyError, TypeError):
        print(json.dumps({"ok": False, "error": "private_receiver_request_failed"}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
