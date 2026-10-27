#!/usr/bin/env bash
# deploy.sh HOST [--gateway FILE] [--adopt] [--no-switch] [--drain-timeout S] [--force] [--settle S] [--keep N]: release
# this repository's commit to HOST as an immutable release (deploy/bin/flatshare-release, ops/RUNBOOK.md's Deploy). The
# release is the commit HEAD names and nothing else (git archive: no untracked or ignored file, no __pycache__, no change
# that is not committed), with DEPLOYED, the commit's `git describe`, which run records name, and on a host that runs
# the gateway (cyan) its binary: FILE, which deploy/build-gateway.sh built from this commit, or, when the crates have not
# changed since the release HOST runs, that release's. A tree with changes the release would not carry is refused. It
# uploads the release to HOST:~/flatshare/releases/.upload-ID (ID, the commit's 12-character abbreviation) and runs
# flatshare-release deploy there, which prepares it (its environments, linked where an earlier release built them, and
# its manifest), verifies it and switches HOST to it, restarting what it changes, and switches back on any failure.
# --adopt moves a host deployed before releases into one first; --no-switch prepares and verifies only.
# deploy.sh HOST --rollback [--to ID]: switch HOST back to its previous release (or ID), by the same steps.
# deploy.sh HOST --status: HOST's releases, current and previous.
# Both run the current release's flatshare-release, or, while the current release is one adopted from before releases
# (which has none), the newest release's.
# HOST is an SSH host name or alias whose first label is the host's directory under deploy/ (phastform, cyan or gray):
# phastform and phastform.example.org both release deploy/phastform/. Set FLATSHARE_SSH to the ssh command for your
# hosts, e.g. FLATSHARE_SSH="ssh -F $HOME/.ssh/flatshare_config".
set -euo pipefail
host=${1:?usage: deploy.sh HOST [--gateway FILE] [--adopt] [--no-switch] [--rollback [--to ID]] [--status] [...]}
shift
name=${host%%.*}
repo=$(cd "$(dirname "$0")/.." && pwd)
ssh=${FLATSHARE_SSH:-ssh}
[[ -d $repo/deploy/$name ]] || { echo "deploy: no deploy/$name directory" >&2; exit 2; }
# The release tool on HOST: the current release's, or the newest release's (by its commit's time, which the files of
# a release keep) where the current release has none.
# shellcheck disable=SC2016 # HOST's shell expands it
tool='t=flatshare/src/deploy/bin/flatshare-release; [ -f "$t" ] ||
  t=$(ls -t flatshare/releases/*/deploy/bin/flatshare-release 2>/dev/null | head -n 1)
  python3 "${t:-flatshare/src/deploy/bin/flatshare-release}"'
gateway="" mode=deploy pass=()
while (($#)); do
  case $1 in
    --gateway) gateway=${2:?--gateway FILE}; shift ;;
    --rollback) mode=rollback ;;
    --status) mode=status ;;
    --adopt | --no-switch | --force) pass+=("$1") ;;
    --to | --drain-timeout | --settle | --keep) pass+=("$1" "${2:?$1 needs a value}"); shift ;;
    *) echo "deploy: unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
if [[ $mode == status ]]; then
  exec $ssh "$host" "$tool status"
fi
if [[ $mode == rollback ]]; then
  exec $ssh "$host" "$tool rollback ${pass[*]:-}"
fi
changes=$(git -C "$repo" status --porcelain --untracked-files=normal)
if [[ -n $changes ]]; then
  echo "deploy: the tree has changes the release would not carry; commit them first:" >&2
  echo "$changes" >&2
  exit 2
fi
# The commit released, which run records name (flatshare-bench e2 run), and the release's id.
rev=$(git -C "$repo" describe --always)
commit=$(git -C "$repo" rev-parse HEAD)
id=$(git -C "$repo" rev-parse --short=12 HEAD)
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
git -C "$repo" archive HEAD | tar -x -C "$stage"
echo "$rev" >"$stage/DEPLOYED"
carry=()
if [[ -e $repo/deploy/$name/systemd/flatshare-gateway.service ]]; then
  if [[ -n $gateway ]]; then
    # A binary deploy/build-gateway.sh built, with its provenance beside it: this commit's, or refused.
    [[ -f $gateway && -f $gateway.json ]] ||
      { echo "deploy: $gateway and $gateway.json (deploy/build-gateway.sh writes both) are needed" >&2; exit 2; }
    built=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["commit"])' "$gateway.json")
    [[ $built == "$commit" ]] ||
      { echo "deploy: $gateway was built from $built, not from this commit, $commit" >&2; exit 2; }
    mkdir -p "$stage/bin"
    cp "$gateway" "$stage/bin/flatshare-gateway"
    chmod 755 "$stage/bin/flatshare-gateway"
    cp "$gateway.json" "$stage/bin/flatshare-gateway.json"
  else
    # The binary of the release HOST runs, when it was built from crates this commit has unchanged: the same source.
    $ssh "$host" "$tool status --json" >"$stage.status" 2>/dev/null || true
    current_id="" current_commit=""
    read -r current_id current_commit < <(python3 - "$stage.status" <<'PY'
import json, sys
try:
    status = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    sys.exit(1)
for release in status["releases"]:
    built = release["artifacts"].get("bin/flatshare-gateway", {}).get("built") or {}
    if release["current"] and built.get("commit"):
        print(release["id"], built["commit"])
PY
    ) || true # none: nothing to carry
    rm -f "$stage.status"
    if [[ -z ${current_commit:-} ]] || ! git -C "$repo" cat-file -e "$current_commit^{commit}" 2>/dev/null ||
      ! git -C "$repo" diff --quiet "$current_commit" HEAD -- crates Cargo.toml Cargo.lock rust-toolchain.toml; then
      echo "deploy: $host runs the gateway, and the release it runs has no binary built from these crates: build" \
        "this commit's (deploy/build-gateway.sh BUILD_HOST) and pass --gateway FILE" >&2
      exit 2
    fi
    carry=(--gateway-from "$current_id")
  fi
fi
tree=$(python3 "$repo/deploy/bin/flatshare-release" tree-hash "$stage")
upload=flatshare/releases/.upload-$id
# An upload left by a deploy that stopped short is made writable again; rsync makes it the stage, whatever it holds.
$ssh "$host" "mkdir -p flatshare/releases && { [ ! -d $upload ] || chmod -R u+w $upload; }"
rsync -a --delete -e "$ssh" "$stage/" "$host:$upload/"
$ssh "$host" "python3 $upload/deploy/bin/flatshare-release deploy $upload --host $name --id $id --commit $commit" \
  "--describe $rev --tree $tree ${carry[*]:-} ${pass[*]:-}"
