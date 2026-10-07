#!/usr/bin/env python3
"""Recover the private ArborScan signing escrow into a new private directory.

Only file paths are arguments. AES-GCM authenticates the complete archive before
any plaintext is written. This tool never replaces existing key material, signs
an APK, changes Git, or claims that a recovered certificate has been verified.
Supported publication platforms: Windows and Linux (renameat2 NOREPLACE).
"""
from __future__ import annotations

import argparse
import base64
import binascii
import ctypes
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile

PREFIX = b"ARBORSCAN-KEYS-1\n"
AAD = b"ArborScan signing material v1"
NONCE_BYTES = 12
TAG_BYTES = 16
MAX_TAR_BYTES = 4 * 1024 * 1024
MAX_ENCRYPTED_BYTES = len(PREFIX) + NONCE_BYTES + MAX_TAR_BYTES + TAG_BYTES
MAX_MEMBER_BYTES = 1024 * 1024
MAX_CREDENTIAL_BYTES = 64 * 1024
MAX_SECRET_BYTES = 1024
FILES = frozenset(("release.p12", "legacy-debug.keystore", "signing.lineage", "credentials.private.json"))


class RecoveryError(RuntimeError):
    """A fixed, safe diagnostic that contains no paths or recovered bytes."""


def _is_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _outside_git(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise RecoveryError("Private recovery paths must be absolute and unambiguous")
    for ancestor in (path, *path.parents):
        if _is_link(ancestor):
            raise RecoveryError("Private recovery paths must not traverse links")
        marker = ancestor / ".git"
        if marker.exists() or _is_link(marker):
            raise RecoveryError("Signing material must remain outside every Git checkout")


def _powershell(script: str) -> str:
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    shell = shutil.which("pwsh.exe") or shutil.which("powershell.exe")
    if shell is None:
        raise RecoveryError("A PowerShell ACL reader is required on Windows")
    environment = os.environ.copy()
    # A PowerShell 7 host can pass incompatible module paths to the built-in
    # Windows PowerShell child. Let that child load its own native ACL module.
    environment.pop("PSModulePath", None)
    try:
        result = subprocess.run(
            [shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            capture_output=True, timeout=30, check=False, env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RecoveryError("Windows private ACL verification could not complete") from None
    if result.returncode:
        raise RecoveryError("Windows private ACL verification could not complete")
    try:
        return result.stdout.decode("utf-8-sig")
    except UnicodeError:
        raise RecoveryError("Windows private ACL verification could not complete") from None


@lru_cache(maxsize=1)
def _current_sid() -> str:
    result = _powershell(
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
        "[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value").strip()
    import re
    if not re.fullmatch(r"S-1-5-[0-9-]+", result):
        raise RecoveryError("Windows identity could not be verified")
    return result


def _windows_private(path: Path) -> None:
    literal = str(path).replace("'", "''")
    result = _powershell(
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
        "$ErrorActionPreference='Stop'; "
        f"$acl=Get-Acl -LiteralPath '{literal}'; "
        "$owner=([System.Security.Principal.NTAccount]$acl.Owner).Translate([System.Security.Principal.SecurityIdentifier]).Value; "
        "$allowed=@($acl.Access | Where-Object {$_.AccessControlType -eq 'Allow'} | "
        "ForEach-Object {$_.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value}); "
        "@{owner=$owner;allowed=$allowed} | ConvertTo-Json -Compress")
    try:
        info = json.loads(result)
        # Python's Windows mkdir(mode=700) can add OWNER RIGHTS. It is safe
        # only because the actual owner is also checked against this whitelist.
        owners = {_current_sid(), "S-1-5-18", "S-1-5-32-544"}
        allowed = owners | {"S-1-3-4"}
        grants = set(info["allowed"])
        if info["owner"] not in owners or not grants or not grants <= allowed:
            raise RecoveryError("Private recovery files and directories must deny access to other accounts")
    except (ValueError, KeyError, TypeError):
        raise RecoveryError("Windows private ACL verification could not complete") from None


def _private(path: Path, directory: bool = False) -> None:
    _outside_git(path)
    try:
        actual = path.lstat()
    except OSError:
        raise RecoveryError("A required private recovery file or directory is unavailable") from None
    if not (stat.S_ISDIR(actual.st_mode) if directory else stat.S_ISREG(actual.st_mode)):
        raise RecoveryError("Only private regular files and directories are permitted")
    if os.name == "nt":
        _windows_private(path)
    elif stat.S_IMODE(actual.st_mode) & 0o077 or (os.geteuid() != 0 and actual.st_uid != os.geteuid()):
        raise RecoveryError("Private recovery files and directories must deny access to other accounts")


def secure_owned_directory(path: Path) -> None:
    """Restrict only a newly created, caller-owned test/staging directory."""
    _outside_git(path)
    if not path.is_dir() or _is_link(path):
        raise RecoveryError("A private staging directory is unavailable")
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["icacls.exe", str(path), "/inheritance:r", "/grant:r",
                 f"*{_current_sid()}:(OI)(CI)F", "*S-1-5-18:(OI)(CI)F", "*S-1-5-32-544:(OI)(CI)F"],
                capture_output=True, timeout=30, check=False, creationflags=subprocess.CREATE_NO_WINDOW)
        except (OSError, subprocess.TimeoutExpired):
            raise RecoveryError("Private staging permissions could not be applied") from None
        if result.returncode:
            raise RecoveryError("Private staging permissions could not be applied")
    else:
        path.chmod(0o700)
    _private(path, directory=True)


def _read_bounded(path: Path, limit: int) -> bytes:
    _private(path)
    _private(path.parent, directory=True)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        with os.fdopen(os.open(path, flags), "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise RecoveryError("Recovery input exceeds its permitted size")
            raw = stream.read(limit + 1)
    except OSError:
        raise RecoveryError("A private recovery input could not be read") from None
    if len(raw) > limit:
        raise RecoveryError("Recovery input exceeds its permitted size")
    return raw


def _decrypt(escrow: bytes, secret: bytes) -> bytes:
    try:
        key = base64.b64decode(secret.strip(), validate=True)
    except (binascii.Error, ValueError):
        raise RecoveryError("The private recovery secret is invalid") from None
    if len(key) != 32:
        raise RecoveryError("The private recovery secret is invalid")
    if not escrow.startswith(PREFIX) or len(escrow) < len(PREFIX) + NONCE_BYTES + TAG_BYTES:
        raise RecoveryError("The encrypted signing escrow format is invalid")
    try:
        from cryptography.exceptions import InvalidTag
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise RecoveryError("Install cryptography in an isolated recovery environment") from None
    nonce = escrow[len(PREFIX):len(PREFIX) + NONCE_BYTES]
    try:
        plain = AESGCM(key).decrypt(nonce, escrow[len(PREFIX) + NONCE_BYTES:], AAD)
    except (InvalidTag, ValueError):
        raise RecoveryError("Signing escrow authentication failed; no material was written") from None
    if len(plain) > MAX_TAR_BYTES:
        raise RecoveryError("The decrypted archive exceeds its permitted size")
    return plain


def _validate_archive(plain: bytes) -> dict[str, bytes]:
    result = {}
    end = 0
    try:
        # Uncompressed only: a compressed or sparse expansion is not permitted.
        with tarfile.open(fileobj=io.BytesIO(plain), mode="r:") as archive:
            for member in archive:
                if (member.name not in FILES or member.name in result or len(result) >= len(FILES)
                        or member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
                        or member.sparse is not None or not 0 < member.size <= MAX_MEMBER_BYTES):
                    raise RecoveryError("The signing archive has an invalid member")
                if member.name == "credentials.private.json" and member.size > MAX_CREDENTIAL_BYTES:
                    raise RecoveryError("Private credential data exceeds its permitted size")
                source = archive.extractfile(member)
                if source is None:
                    raise RecoveryError("The signing archive has an invalid member")
                with source:
                    raw = source.read(member.size + 1)
                if len(raw) != member.size:
                    raise RecoveryError("The signing archive is incomplete")
                result[member.name] = raw
                end = ((member.offset_data + member.size + 511) // 512) * 512
        if set(result) != FILES or len(plain) - end < 1024 or any(plain[end:]):
            raise RecoveryError("The signing archive is incomplete or has unexpected trailing data")
        def no_duplicate_keys(pairs):
            values = {}
            for name, value in pairs:
                if name in values:
                    raise RecoveryError("Private credentials contain duplicate fields")
                values[name] = value
            return values
        credentials = json.loads(result["credentials.private.json"].decode("utf-8"),
                                 object_pairs_hook=no_duplicate_keys)
        if (not isinstance(credentials, dict) or set(credentials) != {"release_password"}
                or not isinstance(credentials["release_password"], str)
                or not 0 < len(credentials["release_password"]) <= 4096
                or "\x00" in credentials["release_password"]):
            raise RecoveryError("Private credentials do not match signing escrow format 1")
    except (tarfile.TarError, UnicodeError, json.JSONDecodeError, OSError, ValueError, KeyError, RecursionError):
        raise RecoveryError("The signing archive or credentials are invalid") from None
    return result


def _write_private(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        with os.fdopen(os.open(path, flags, 0o600), "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        _private(path)
    except OSError:
        raise RecoveryError("A private staging file could not be written") from None


def _publish(stage: Path, destination: Path) -> None:
    # Windows rename refuses an existing destination. Linux needs NOREPLACE:
    # plain rename could silently replace an empty destination in a race.
    if os.name == "nt":
        try:
            os.rename(stage, destination)
        except OSError:
            raise RecoveryError("Recovery publication refused or did not complete") from None
    elif sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        rename = getattr(library, "renameat2", None)
        if rename is None:
            raise RecoveryError("Atomic no-replace directory publication is unavailable")
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(stage), -100, os.fsencode(destination), 1):
            raise RecoveryError("Recovery publication refused or did not complete")
    else:
        raise RecoveryError("Atomic recovery publication is supported on Windows and Linux")


def _verify_staged(stage: Path, files: dict[str, bytes]) -> None:
    if {path.name for path in stage.iterdir()} != FILES:
        raise RecoveryError("Private staging contains unexpected material")
    for name, expected in files.items():
        path = stage / name
        if _is_link(path) or not path.is_file():
            raise RecoveryError("Private staging contains an invalid file")
        with path.open("rb") as stream:
            raw = stream.read(len(expected) + 1)
            if len(raw) != len(expected) or hashlib.sha256(raw).digest() != hashlib.sha256(expected).digest():
                raise RecoveryError("Private staged content did not pass verification")


def _cleanup_owned_stage(stage: Path, parent: Path) -> None:
    if stage.exists():
        if (_is_link(stage) or stage.parent.resolve() != parent.resolve()
                or not stage.name.startswith(".signing-recovery-")):
            raise RecoveryError("Suspicious staging was retained; no recursive cleanup was attempted")
        try:
            shutil.rmtree(stage)
        except OSError:
            raise RecoveryError("Private staging cleanup could not complete") from None


def recover(escrow: Path, secret: Path, destination: Path) -> dict:
    escrow, secret, destination = Path(escrow), Path(secret), Path(destination)
    _outside_git(destination)
    if destination.exists() or _is_link(destination):
        raise RecoveryError("Recovery destination must be new; existing material is never replaced")
    _private(destination.parent, directory=True)
    parent_identity = destination.parent.stat()
    # Both authentication and all archive validation precede plaintext staging.
    encrypted = _read_bounded(escrow, MAX_ENCRYPTED_BYTES)
    key = _read_bounded(secret, MAX_SECRET_BYTES)
    files = _validate_archive(_decrypt(encrypted, key))
    stage = None
    try:
        stage = Path(tempfile.mkdtemp(prefix=".signing-recovery-", dir=destination.parent))
        secure_owned_directory(stage)
        for name in sorted(FILES):
            _write_private(stage / name, files[name])
        _verify_staged(stage, files)
        _outside_git(destination)
        _private(destination.parent, directory=True)
        after = destination.parent.stat()
        if (parent_identity.st_dev, parent_identity.st_ino) != (after.st_dev, after.st_ino):
            raise RecoveryError("The private destination parent changed during recovery")
        if destination.exists() or _is_link(destination):
            raise RecoveryError("Recovery destination must be new; existing material is never replaced")
        _publish(stage, destination)
        stage = None
    finally:
        if stage is not None:
            _cleanup_owned_stage(stage, destination.parent)
    return {"status": "recovered_private_signing_material", "files_recovered": len(FILES),
            "escrow_sha256": hashlib.sha256(encrypted).hexdigest(),
            "existing_keys_replaced": False, "certificate_verified": False,
            "apk_signed": False}


class _SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise RecoveryError("Invalid recovery arguments; use --help")


def main(argv=None) -> int:
    parser = _SafeParser(description=__doc__)
    parser.add_argument("--escrow", type=Path, required=True, help="Absolute path to the private encrypted escrow")
    parser.add_argument("--recovery-secret-file", type=Path, required=True, help="Absolute path to the private base64 key file")
    parser.add_argument("--destination", type=Path, required=True, help="New directory beneath an existing private parent, outside Git")
    try:
        args = parser.parse_args(argv)
        print(json.dumps(recover(args.escrow, args.recovery_secret_file, args.destination)))
        return 0
    except RecoveryError as error:
        # RecoveryError messages are fixed literals, without user-provided data.
        print(json.dumps({"status": "failed", "reason": str(error)}))
        return 1
    except OSError:
        print(json.dumps({"status": "failed", "reason": "private_signing_recovery_not_confirmed"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
