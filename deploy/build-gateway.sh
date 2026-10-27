#!/usr/bin/env bash
# build-gateway.sh BUILD_HOST [OUT]: build this repository's commit of flatshare-gateway on BUILD_HOST, a Linux host with
# rustup whose glibc is not newer than the control host's (peacock and cyan run Ubuntu 22.04), and copy it to OUT
# (default build/flatshare-gateway-ID, ID the commit's 12-character abbreviation) beside OUT.json, its provenance: the
# commit and its `git describe`, the build host, the toolchain (rust-toolchain.toml's, as rustc names itself), the
# SHA-256 of Cargo.lock and of the binary, and its --version. The build is `git archive HEAD` in a directory of its
# own, `cargo build --release --locked -p flatshare-gateway`, with a target directory kept across builds; a tree with
# changes that the archive would not carry is refused, as deploy.sh refuses one. deploy.sh cyan --gateway OUT puts the
# binary in cyan's release, and refuses one built from another commit. FLATSHARE_SSH is the ssh command, as for
# deploy.sh.
set -euo pipefail
build_host=${1:?usage: build-gateway.sh BUILD_HOST [OUT]}
repo=$(cd "$(dirname "$0")/.." && pwd)
ssh=${FLATSHARE_SSH:-ssh}
changes=$(git -C "$repo" status --porcelain --untracked-files=normal)
if [[ -n $changes ]]; then
  echo "build-gateway: the tree has changes the build would not carry; commit them first:" >&2
  echo "$changes" >&2
  exit 2
fi
commit=$(git -C "$repo" rev-parse HEAD)
id=$(git -C "$repo" rev-parse --short=12 HEAD)
out=${2:-$repo/build/flatshare-gateway-$id}
mkdir -p "$(dirname "$out")"
work=flatshare-gateway-build
git -C "$repo" archive HEAD | $ssh "$build_host" "rm -rf $work/src && mkdir -p $work/src && tar -x -C $work/src"
$ssh "$build_host" "cd $work/src && export PATH=\$HOME/.cargo/bin:\$PATH && CARGO_TARGET_DIR=\$HOME/$work/target \
  cargo build -q --release --locked -p flatshare-gateway"
toolchain=$($ssh "$build_host" "cd $work/src && export PATH=\$HOME/.cargo/bin:\$PATH && rustc --version")
$ssh "$build_host" "cat $work/target/release/flatshare-gateway" >"$out.new"
chmod 755 "$out.new"
mv "$out.new" "$out"
version=$($ssh "$build_host" "$work/target/release/flatshare-gateway --version")
python3 - "$out" "$commit" "$(git -C "$repo" describe --always)" "$build_host" "$toolchain" "$version" \
  "$repo/Cargo.lock" <<'PY'
import hashlib, json, sys
out, commit, describe, host, toolchain, version, lock = sys.argv[1:]
sha = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()
provenance = {
    "commit": commit,
    "describe": describe,
    "build_host": host,
    "toolchain": toolchain,
    "command": "cargo build --release --locked -p flatshare-gateway",
    "cargo_lock_sha256": sha(lock),
    "sha256": sha(out),
    "version": version,
}
with open(out + ".json", "w") as f:
    f.write(json.dumps(provenance, indent=1) + "\n")
print(f"build-gateway: {out}: {version}, sha256 {provenance['sha256'][:16]}, built on {host} from {describe}")
PY
