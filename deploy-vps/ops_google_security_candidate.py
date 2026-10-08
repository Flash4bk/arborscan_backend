"""Prepare a security-only Docker build context; never deploy or access a DB.

The caller must privately capture server.py from the inspected live image, check
its SHA, and tag that same immutable image with base_tag(image_id). Before and
after `docker build --pull=false`, independently verify that tag still names
image_id. This utility does not verify Docker's local tag mapping or grant
permission to update API v3. Windows output inherits the private parent ACL;
choose a private output parent. No production credentials belong in this context.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


PREPARED_COMMIT = "6284f93204f3ef2f92543604eb5c7a2cea93c3cd"
PREPARED_SHA = {
    "server.py": "5288a9d23db46b362db84b96957d7bf91091b91155c6837da6b97e6b9a136d2c",
    "auth_google_identity.py": "451b1273a68ecba63253eb12c48bc1a35c159d41329336234e60b4b6267b3bec",
}
BASELINE_AUTH_SHA = "e96d3a12d9bf834f16c3f0dc046fa1dd77d3ec8cafd54a87e1404404ddce6a31"
DIGEST = re.compile(r"[0-9a-f]{64}")
FLAG = "GOOGLE_IDENTITY_CLAIMS_ENABLED"
LIMIT = 10 * 1024 * 1024


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def base_tag(image_id):
    require(isinstance(image_id, str) and image_id.startswith("sha256:") and
            DIGEST.fullmatch(image_id[7:]), "invalid_base_image_id")
    return "arborscan-google-security-base:sha256-" + image_id[7:]


def no_links(path):
    for item in (path, *path.parents):
        require(not item.is_symlink() and not getattr(item, "is_junction", lambda: False)(),
                "symlink_or_junction_not_allowed")


def read_source(path):
    no_links(path)
    require(path.is_file() and path.stat().st_size <= LIMIT, "source_missing_or_oversized")
    data = path.read_bytes()
    require(b"\x00" not in data, "invalid_source_encoding")
    data.decode("utf-8")
    return data


def prepared_sources(repo):
    """Read pinned Git objects and reject a modified checkout policy/source."""
    result = {}
    for name, expected in PREPARED_SHA.items():
        process = subprocess.run(["git", "show", PREPARED_COMMIT + ":" + name],
                                 cwd=repo, capture_output=True, timeout=20, check=False)
        require(process.returncode == 0, "prepared_git_object_unavailable")
        data = process.stdout
        require(sha(data) == expected, "prepared_git_object_digest_mismatch")
        require(read_source(repo / name).replace(b"\r\n", b"\n") == data,
                "prepared_checkout_modified")
        result[name] = data
    return result


def one(nodes, predicate, reason):
    selected = [node for node in nodes if predicate(node)]
    require(len(selected) == 1, reason)
    return selected[0]


def is_flag(node):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == FLAG for target in node.targets)


def auth_node(tree):
    node = one(tree.body, lambda n: isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == "auth_google", "ambiguous_auth_function")
    require(isinstance(node, ast.AsyncFunctionDef), "unexpected_auth_function_type")
    routes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and any(isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                      and isinstance(d.func.value, ast.Name) and d.func.value.id == "app"
                      and d.func.attr == "post" and d.args
                      and isinstance(d.args[0], ast.Constant)
                      and d.args[0].value == "/auth/google" for d in n.decorator_list)]
    require(routes == [node] and len(node.decorator_list) == 1,
            "ambiguous_google_route")
    return node


def span(data, node):
    lines = data.splitlines(keepends=True)
    return (sum(map(len, lines[:node.lineno - 1])) + node.col_offset,
            sum(map(len, lines[:node.end_lineno - 1])) + node.end_col_offset)


def segment(data, node):
    start, end = span(data, node)
    return data[start:end]


def transform(baseline, expected_sha, prepared):
    require(isinstance(expected_sha, str) and DIGEST.fullmatch(expected_sha),
            "invalid_baseline_digest")
    require(sha(baseline) == expected_sha, "baseline_digest_mismatch")
    for name, expected in PREPARED_SHA.items():
        require(name in prepared and sha(prepared[name]) == expected,
                "prepared_source_digest_mismatch")
    tree = ast.parse(baseline.decode("utf-8"))
    fixed_tree = ast.parse(prepared["server.py"].decode("utf-8"))
    old, fixed = auth_node(tree), auth_node(fixed_tree)
    require(sha(segment(baseline, old)) == BASELINE_AUTH_SHA, "unexpected_baseline_auth_digest")
    require(not any(isinstance(n, ast.ImportFrom) and n.module == "auth_google_identity"
                    for n in ast.walk(tree)), "policy_already_present")
    require(not any(isinstance(n, ast.Name) and n.id == FLAG for n in ast.walk(tree)),
            "claims_flag_already_present")
    anchor = one(tree.body, lambda n: isinstance(n, ast.ImportFrom) and n.module == "config"
                 and len(n.names) == 1 and n.names[0].name == "settings"
                 and n.names[0].asname is None, "ambiguous_config_anchor")
    require(any(isinstance(n, ast.Import) and any(a.name == "os" and a.asname is None
                for a in n.names) and n.lineno < anchor.lineno for n in tree.body),
            "os_import_unavailable_at_anchor")
    policy = one(fixed_tree.body, lambda n: isinstance(n, ast.ImportFrom)
                 and n.module == "auth_google_identity", "prepared_policy_import_ambiguous")
    flag = one(fixed_tree.body, is_flag, "prepared_claims_flag_ambiguous")
    require([ast.dump(d) for d in old.decorator_list] ==
            [ast.dump(d) for d in fixed.decorator_list], "route_decorator_mismatch")
    newline = b"\r\n" if b"\r\n" in baseline else b"\n"
    adapt = lambda value: value.replace(b"\r\n", b"\n").replace(b"\n", newline)
    insertion = newline + adapt(segment(prepared["server.py"], policy)) + newline + \
        adapt(segment(prepared["server.py"], flag)) + newline
    anchor_end = span(baseline, anchor)[1]
    start, end = span(baseline, old)
    require(anchor_end < start, "config_anchor_after_auth_function")
    replacement = adapt(segment(prepared["server.py"], fixed))
    candidate = (baseline[:anchor_end] + insertion + baseline[anchor_end:start] +
                 replacement + baseline[end:])
    candidate_tree = ast.parse(candidate.decode("utf-8"))
    unchanged = [n for n in candidate_tree.body if not (
        isinstance(n, ast.ImportFrom) and n.module == "auth_google_identity") and not is_flag(n)]
    for position, node in enumerate(unchanged):
        if node is auth_node(candidate_tree):
            unchanged[position] = old
    require(ast.dump(ast.Module(body=unchanged, type_ignores=[])) ==
            ast.dump(ast.Module(body=tree.body, type_ignores=[])), "unexpected_ast_change")
    require(ast.dump(auth_node(candidate_tree)) == ast.dump(fixed), "candidate_route_mismatch")
    # The three slices contain every baseline byte outside auth_google. They are
    # copied directly above, without formatting or reserialization of the server.
    preserved = baseline[:anchor_end] + baseline[anchor_end:start] + baseline[end:]
    return candidate, sha(preserved)


def prepare(repo, baseline_path, baseline_sha, image_id, output):
    repo, baseline_path, output = Path(repo), Path(baseline_path), Path(output)
    require(repo.is_absolute() and baseline_path.is_absolute() and output.is_absolute(),
            "absolute_paths_required")
    require(".." not in output.parts, "unsafe_output_path")
    no_links(output)
    require(output.parent.is_dir() and not output.exists(), "output_must_be_new_directory")
    require(not output.resolve().is_relative_to(repo.resolve()), "output_inside_source_repository")
    tag = base_tag(image_id)
    baseline = read_source(baseline_path)
    prepared = prepared_sources(repo)
    candidate, preserved_sha = transform(baseline, baseline_sha, prepared)
    dockerfile = ("# Security overlay only. Verify local tag -> manifest image ID before/after build.\n"
                  f"FROM {tag}\n"
                  "COPY --chown=1000:1000 server.py auth_google_identity.py /app/\n"
                  "ENV ARBORSCAN_GOOGLE_IDENTITY_CLAIMS_ENABLED=false\n").encode("ascii")
    files = {"server.py": candidate, "auth_google_identity.py": prepared["auth_google_identity.py"],
             "Dockerfile": dockerfile}
    manifest = {"format": 1, "purpose": "as16_google_security_candidate_not_deployed",
                "prepared_commit": PREPARED_COMMIT, "baseline_server_sha256": sha(baseline),
                "baseline_auth_sha256": BASELINE_AUTH_SHA,
                "unchanged_baseline_bytes_sha256": preserved_sha,
                "base_image_id": image_id, "base_local_tag": tag,
                "base_tag_mapping_verified_by_this_tool": False,
                "required_build_guards": ["verify_local_tag_equals_base_image_id_before_build",
                                          "docker_build_pull_false",
                                          "verify_local_tag_equals_base_image_id_after_build"],
                "identity_claims_default_enabled": False,
                "production_mutations": False,
                "file_sha256": {name: sha(data) for name, data in files.items()}}
    files["provenance.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.mkdir(mode=0o700)
    for name, data in files.items():
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(output / name, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--base-image-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        manifest = prepare(args.repo, args.baseline, args.baseline_sha256,
                           args.base_image_id, args.output)
    except (ValueError, OSError, SyntaxError, UnicodeError, subprocess.SubprocessError):
        # Exception details can contain source fragments, private paths or Git
        # stderr; the CLI deliberately emits only this safe failure code.
        parser.exit(1, "candidate_preparation_refused\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
