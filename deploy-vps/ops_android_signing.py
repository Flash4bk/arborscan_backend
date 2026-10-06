#!/usr/bin/env python3
"""Private Android APK signing, SDK-targeted rotation, and public evidence.

Gradle builds an unsigned, non-debuggable release. This tool then signs it with
the old signer below API 33 and the permanent signer on API 33+, preserving the
installed application's data. No password may appear in configuration as text,
command arguments, logs, or public evidence: use file:/env: password sources.

Official references (reviewed 2026-10-06):
https://developer.android.com/tools/apksigner
https://source.android.com/docs/security/features/apksigning/v3
https://source.android.com/docs/security/features/apksigning/v3-1
https://developer.android.com/studio/publish/app-signing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Callable


class SigningError(RuntimeError):
    """A safe error whose message never contains secret tool output."""


REPOSITORY = Path(__file__).resolve().parent.parent
PLATFORMS = ((24, 27, "old", "v2"), (28, 32, "old", "v3"),
             (33, 33, "new", "v3.1"), (36, 36, "new", "v3.1"))


def fingerprint(value: str, length: int = 64) -> str:
    normalized = str(value).replace(":", "").lower()
    if not re.fullmatch(r"[0-9a-f]{%d}" % length, normalized):
        raise SigningError("Invalid public certificate fingerprint")
    return normalized


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def outside_repository(path: Path) -> None:
    resolved = path.resolve()
    if (resolved == REPOSITORY or REPOSITORY in resolved.parents
            or any((parent / ".git").exists() for parent in resolved.parents)):
        raise SigningError("Signing material must be outside the repository")


def private_file(path: Path) -> Path:
    if not path.is_absolute():
        raise SigningError("Private file paths must be absolute")
    outside_repository(path)
    if not path.is_file() or path.is_symlink():
        raise SigningError("A required private regular file is unavailable")
    if os.name != "nt" and path.stat().st_mode & 0o077:
        raise SigningError("Private files must deny group/other access")
    # Windows ACLs are configured and audited separately, never inferred from
    # POSIX mode bits. This tool does not weaken or replace those ACLs.
    return path.resolve()


def load_configuration(path: Path) -> dict:
    path = private_file(path)
    try:
        config = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise SigningError("Private signing configuration cannot be read") from exc
    allowed = {"format_version", "application_id", "min_sdk", "rotation_min_sdk",
               "build_tools", "java_home", "lineage", "old_signer", "new_signer"}
    if not isinstance(config, dict) or set(config) != allowed:
        raise SigningError("Private configuration fields do not match format 1")
    if (config["format_version"] != 1
            or config["application_id"] != "com.example.arborscan_app"
            or config["min_sdk"] != 24 or config["rotation_min_sdk"] != 33):
        raise SigningError("Unsupported application or Android compatibility policy")
    for name in ("build_tools", "java_home", "lineage"):
        value = Path(config[name])
        if not value.is_absolute():
            raise SigningError("Signing tool and lineage paths must be absolute")
    outside_repository(Path(config["lineage"]))
    for name in ("old_signer", "new_signer"):
        signer = config[name]
        if not isinstance(signer, dict) or set(signer) != {
                "keystore", "alias", "store_password", "key_password", "sha256", "sha1"}:
            raise SigningError("Invalid signer configuration")
        private_file(Path(signer["keystore"]))
        if not isinstance(signer["alias"], str) or not signer["alias"]:
            raise SigningError("A signer alias is required")
        for key in ("store_password", "key_password"):
            source = signer[key]
            if not isinstance(source, str) or ":" not in source:
                raise SigningError("Passwords must use file: or env: sources")
            kind, value = source.split(":", 1)
            if kind == "file":
                private_file(Path(value))
            elif kind == "env":
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) or not os.environ.get(value):
                    raise SigningError("A required private password environment variable is unavailable")
            else:
                raise SigningError("Inline passwords are prohibited")
        signer["sha256"] = fingerprint(signer["sha256"])
        signer["sha1"] = fingerprint(signer["sha1"], 40)
    if config["old_signer"]["sha256"] == config["new_signer"]["sha256"]:
        raise SigningError("The permanent signer must differ from the old signer")
    return config


def execute(config: dict, name: str, args: list[str], operation: str) -> str:
    suffix = ".bat" if name == "apksigner" and os.name == "nt" else ".exe" if os.name == "nt" else ""
    tool = Path(config["build_tools"]) / (name + suffix)
    if not tool.is_file():
        raise SigningError("A required Android build tool is unavailable")
    environment = os.environ.copy()
    environment["JAVA_HOME"] = config["java_home"]
    try:
        result = subprocess.run([str(tool), *args], capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                env=environment, check=False, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SigningError(f"{operation} could not complete") from exc
    if result.returncode:
        # Tool output and arguments can contain private paths or provider
        # diagnostics. Retain neither in public logs nor in exception messages.
        raise SigningError(f"{operation} failed (exit {result.returncode})")
    return result.stdout


def signer_arguments(signer: dict) -> list[str]:
    return ["--ks", signer["keystore"], "--ks-key-alias", signer["alias"],
            "--ks-pass", signer["store_password"], "--key-pass", signer["key_password"]]


def normalize_tool_output(output: str) -> str:
    # Windows SDK .bat wrappers and retained diagnostics can have CRLF, or an
    # extra CR before LF. Regex line anchors must not mistake that for another
    # signer/certificate. Keep all actual certificate values unchanged.
    return output.replace("\r\n", "\n").replace("\r", "\n")


def parse_manifest(output: str) -> dict:
    output = normalize_tool_output(output)
    package = re.search(r"^package: name='([^']+)' versionCode='(\d+)' versionName='([^']+)'", output, re.M)
    minimum = re.search(r"^sdkVersion:'(\d+)'", output, re.M)
    target = re.search(r"^targetSdkVersion:'(\d+)'", output, re.M)
    permissions = re.findall(r"^uses-permission: name='([^']+)'", output, re.M)
    if not package or not minimum or not target:
        raise SigningError("APK manifest metadata is incomplete")
    return {"application_id": package[1], "version_code": int(package[2]),
            "version_name": package[3], "min_sdk": int(minimum[1]),
            "target_sdk": int(target[1]), "debuggable": bool(re.search(
                r"^application-debuggable(?:\s|$)", output, re.M)),
            "internet_permission": "android.permission.INTERNET" in permissions}


def parse_signatures(output: str, minimum_sdk: int | None = None,
                     maximum_sdk: int | None = None) -> dict:
    output = normalize_tool_output(output)
    label = r"Signer (?:#\d+|\(minSdkVersion=\d+, maxSdkVersion=\d+\))"
    certificates = re.findall(r"^(" + label + r") certificate SHA-256 digest: ([0-9a-fA-F]+)$", output, re.M)
    certificates_sha1 = dict(re.findall(r"^(" + label + r") certificate SHA-1 digest: ([0-9a-fA-F]+)$", output, re.M))
    schemes = {name: value == "true" for name, value in re.findall(
        r"^Verified using (v\d(?:\.\d)?) scheme [^\n]*: (true|false)$", output, re.M)}
    signer_count = re.search(r"^Number of signers: (\d+)$", output, re.M)
    if signer_count and signer_count[1] != "1":
        raise SigningError("Expected a single signing identity")
    targeted = [row for row in certificates if "minSdkVersion=" in row[0]]
    if targeted:
        if len(targeted) != len(certificates) or minimum_sdk is None or maximum_sdk is None:
            raise SigningError("SDK-targeted certificate output requires an explicit SDK range")
        effective = []
        for row in targeted:
            limits = re.search(r"minSdkVersion=(\d+), maxSdkVersion=(\d+)", row[0])
            if int(limits[1]) <= minimum_sdk and maximum_sdk <= int(limits[2]):
                effective.append(row)
        certificates = effective
    if len(certificates) != 1 or certificates[0][0] not in certificates_sha1:
        raise SigningError("Expected exactly one effective APK signer for this SDK range")
    selected, certificate = certificates[0]
    return {"sha256": fingerprint(certificate),
            "sha1": fingerprint(certificates_sha1[selected], 40), "schemes": schemes}


def verify_lineage(config: dict, path: Path, runner: Callable = execute) -> list[dict]:
    output = runner(config, "apksigner", ["lineage", "--in", str(path), "--print-certs"],
                    "Signing lineage inspection")
    output = normalize_tool_output(output)
    blocks = re.split(r"(?=^Signer #\d+ in lineage certificate DN:)", output, flags=re.M)
    blocks = [block for block in blocks if re.match(r"Signer #\d+ in lineage certificate DN:", block)]
    if len(blocks) != 2:
        raise SigningError("Expected an exact old-to-permanent two-certificate lineage")
    result = []
    for role, block in zip(("old", "new"), blocks):
        certificate = re.search(r"certificate SHA-256 digest: ([0-9a-fA-F]+)", block)
        certificate_sha1 = re.search(r"certificate SHA-1 digest: ([0-9a-fA-F]+)", block)
        data = re.search(r"Has installed data capability\s*:\s*(true|false)", block)
        rollback = re.search(r"Has rollback capability\s*:\s*(true|false)", block)
        if not certificate or not certificate_sha1 or not data or not rollback:
            raise SigningError("Lineage certificate capabilities are incomplete")
        if (fingerprint(certificate[1]) != config[f"{role}_signer"]["sha256"]
                or fingerprint(certificate_sha1[1], 40) != config[f"{role}_signer"]["sha1"]):
            raise SigningError("Lineage certificates do not match the configured key transition")
        if role == "old" and (data[1] != "true" or rollback[1] != "false"):
            raise SigningError("Lineage must preserve installed data and prohibit old-key rollback")
        result.append({"signer_role": role, "certificate_sha256": fingerprint(certificate[1]),
                       "installed_data": data[1] == "true", "rollback": rollback[1] == "true"})
    return result


def verify_apk(config: dict, apk: Path, minimum_version_code: int,
               runner: Callable = execute, *, instrumentation: bool = False) -> dict:
    if not apk.is_file() or apk.is_symlink():
        raise SigningError("APK must be an existing regular file")
    manifest = parse_manifest(runner(config, "aapt", ["dump", "badging", str(apk)], "APK manifest inspection"))
    expected_id = config["application_id"] + (".test" if instrumentation else "")
    if (manifest["application_id"] != expected_id
            or manifest["min_sdk"] != config["min_sdk"] or manifest["target_sdk"] != 36
            or manifest["version_code"] < minimum_version_code):
        raise SigningError("APK application, SDK range, or versionCode is incompatible")
    if manifest["debuggable"] and not instrumentation:
        raise SigningError("A release APK must not be debuggable")
    if not manifest["internet_permission"] and not instrumentation:
        raise SigningError("A release APK requires the INTERNET permission")
    if instrumentation:
        xml = runner(config, "aapt", ["dump", "xmltree", str(apk), "AndroidManifest.xml"],
                     "Instrumentation target inspection")
        if not (re.search(r'android:targetPackage[^\n]*="' + re.escape(config["application_id"]) + r'"', xml)
                and re.search(r'android:name[^\n]*="com\.example\.arborscan_app\.DataPreservationInstrumentation"', xml)):
            raise SigningError("Test APK must target the ArborScan preservation audit runner")
    platforms = []
    for minimum, maximum, signer, scheme in PLATFORMS:
        result = parse_signatures(runner(config, "apksigner", ["verify", "--verbose", "--print-certs",
            "--min-sdk-version", str(minimum), "--max-sdk-version", str(maximum), str(apk)],
            f"APK signature verification API {minimum}-{maximum}"), minimum, maximum)
        expected = config[f"{signer}_signer"]["sha256"]
        if (result["sha256"] != expected
                or result["sha1"] != config[f"{signer}_signer"]["sha1"]
                or not result["schemes"].get(scheme)):
            raise SigningError(f"APK signer or scheme does not match API {minimum}-{maximum} policy")
        platforms.append({"min_sdk": minimum, "max_sdk": maximum,
                          "signer_role": signer, **result})
    runner(config, "zipalign", ["-c", "-P", "16", "-v", "4", str(apk)], "APK alignment verification")
    lineage = private_file(Path(config["lineage"]))
    lineage_capabilities = verify_lineage(config, lineage, runner)
    if verify_lineage(config, apk, runner) != lineage_capabilities:
        raise SigningError("APK embedded lineage differs from the configured transition")
    return {"format_version": 1,
            "artifact_kind": "instrumentation" if instrumentation else "application",
            "apk_sha256": sha256(apk), "apk_bytes": apk.stat().st_size,
            "manifest": manifest, "rotation_min_sdk": config["rotation_min_sdk"],
            "lineage_sha256": sha256(lineage), "platform_signatures": platforms,
            "lineage_capabilities": lineage_capabilities,
            "old_certificate_sha1": config["old_signer"]["sha1"],
            "new_certificate_sha1": config["new_signer"]["sha1"],
            "device_installation_verified": False}


def no_existing_output(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise SigningError("Refusing to overwrite an existing output")
    path.parent.mkdir(parents=True, exist_ok=True)


def rotate(config: dict, runner: Callable = execute) -> None:
    lineage = Path(config["lineage"])
    outside_repository(lineage)
    no_existing_output(lineage)
    # installed-data is necessary for an in-place update. Rollback to the
    # development key is deliberately disabled; forward-fix with the new key.
    with tempfile.TemporaryDirectory(prefix=".rotation-", dir=lineage.parent) as stage:
        temporary = Path(stage) / "lineage"
        runner(config, "apksigner", ["rotate", "--out", str(temporary), "--old-signer",
            *signer_arguments(config["old_signer"]), "--set-installed-data", "true",
            "--set-rollback", "false", "--new-signer",
            *signer_arguments(config["new_signer"])], "Signing lineage creation")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise SigningError("Signing lineage was not created")
        verify_lineage(config, temporary, runner)
        temporary.chmod(0o600)
        # Atomically publish the complete file without overwriting a lineage
        # created concurrently; staging is on the same filesystem.
        os.link(temporary, lineage)


def sign(config: dict, source: Path, output: Path, minimum_version_code: int,
         runner: Callable = execute, *, instrumentation: bool = False) -> dict:
    if not source.is_file() or source.is_symlink():
        raise SigningError("Unsigned APK is unavailable")
    private_file(Path(config["lineage"]))
    verify_lineage(config, Path(config["lineage"]), runner)
    no_existing_output(output)
    with tempfile.TemporaryDirectory(prefix=".apk-signing-", dir=output.parent) as stage:
        aligned = Path(stage) / "aligned.apk"
        signed = Path(stage) / "signed.apk"
        runner(config, "zipalign", ["-P", "16", "-f", "4", str(source), str(aligned)], "APK alignment")
        runner(config, "apksigner", ["sign", "--min-sdk-version", str(config["min_sdk"]),
            "--rotation-min-sdk-version", str(config["rotation_min_sdk"]),
            "--v1-signing-enabled", "false", "--v2-signing-enabled", "true",
            "--v3-signing-enabled", "true", "--v4-signing-enabled", "false",
            "--lineage", config["lineage"], "--out", str(signed),
            *signer_arguments(config["old_signer"]), "--next-signer",
            *signer_arguments(config["new_signer"]), str(aligned)], "APK signing")
        metadata = verify_apk(config, signed, minimum_version_code, runner,
                              instrumentation=instrumentation)
        os.link(signed, output)
        if sha256(output) != metadata["apk_sha256"]:
            raise SigningError("Published APK checksum does not match verified staged APK")
        return metadata


def write_public_metadata(path: Path, metadata: dict) -> None:
    no_existing_output(path)
    # Only explicit whitelisted public fields from verify_apk reach this file.
    with tempfile.TemporaryDirectory(prefix=".apk-evidence-", dir=path.parent) as stage:
        staged = Path(stage) / "metadata.json"
        with staged.open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.link(staged, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("rotate", "sign", "verify"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--apk", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--minimum-version-code", type=int)
    parser.add_argument("--instrumentation", action="store_true",
                        help="Sign/verify the restricted .test audit APK, never a production APK")
    arguments = parser.parse_args()
    try:
        config = load_configuration(arguments.config)
        if arguments.command == "rotate":
            rotate(config)
            print("Signing lineage created; existing key material was not replaced.")
            return 0
        minimum_version = arguments.minimum_version_code
        if minimum_version is None:
            minimum_version = 1 if arguments.instrumentation else 16
        if minimum_version < (1 if arguments.instrumentation else 16):
            raise SigningError("Release versionCode must be greater than the previous debug candidate")
        if arguments.metadata:
            no_existing_output(arguments.metadata)
        if arguments.command == "sign":
            if arguments.input is None or arguments.output is None:
                raise SigningError("Sign requires --input and --output")
            metadata = sign(config, arguments.input.resolve(), arguments.output.resolve(), minimum_version,
                            instrumentation=arguments.instrumentation)
        else:
            if arguments.apk is None:
                raise SigningError("Verify requires --apk")
            metadata = verify_apk(config, arguments.apk.resolve(), minimum_version,
                                  instrumentation=arguments.instrumentation)
        if arguments.metadata:
            write_public_metadata(arguments.metadata, metadata)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    except SigningError as exc:
        print(f"Signing operation refused: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError):
        print("Signing operation refused: invalid or unavailable private input/output", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
