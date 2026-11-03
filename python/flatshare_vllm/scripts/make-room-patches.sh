#!/usr/bin/env bash
# make-room-patches.sh VLLM_CLONE: write make-room's three patches (src/flatshare_vllm/make-room/*.patch) from the
# backport's branches, which the fork https://github.com/StevenWang-CY/vllm publishes. VLLM_CLONE is a clone of the
# fork with vLLM's release tag fetched into it, which the fork does not carry
# (git -C VLLM_CLONE fetch https://github.com/vllm-project/vllm tag v0.32.0). Each branch is the release with commits
# on top, and each patch is the scheduler.py part of one branch's commits. Its header names the commits that change
# scheduler.py and, under a commit that the release made differ from its PR commit, the paragraph of its message that
# says how ("Backport: ..."):
#   v0.32.0-priority-kv-preemption   v0.32.0 + #65282's commit, cherry-picked                   -> 63350.patch
#   v0.32.0-priority-prefill-hold    v0.32.0-priority-kv-preemption + #65283's own commit        -> 63351.patch
#   v0.32.0-flatshare-make-room      v0.32.0-priority-prefill-hold + the variant's commits       -> variant.patch
# (The patches keep the names of the pull requests they were first written for, #63350 and #63351, which #65282
# and #65283 superseded on 2026-11-03.)
# A change to the PRs is made on their branches (the fork's priority-kv-preemption and priority-prefill-hold, on
# vLLM main), cherry-picked onto these and pushed to the fork, and the patches rewritten here from the clone's copies
# of the fork's branches (origin/...). KV_REF, CAP_REF and VARIANT_REF name other branches. BASE names another release
# (BASE=v0.33.0: the fork's v0.33.0-* branches, the same commits cherry-picked onto v0.33.0), whose patches are written
# to make-room/BASE/.
set -euo pipefail
clone=${1:?usage: make-room-patches.sh VLLM_CLONE (a clone of https://github.com/StevenWang-CY/vllm)}
base=${BASE:-v0.32.0}
kv=${KV_REF:-origin/$base-priority-kv-preemption}
cap=${CAP_REF:-origin/$base-priority-prefill-hold}
variant=${VARIANT_REF:-origin/$base-flatshare-make-room}
out=$(cd "$(dirname "$0")/../src/flatshare_vllm/make-room" && pwd)
[[ $base == v0.32.0 ]] || { out=$out/$base; mkdir -p "$out"; }
scheduler=vllm/v1/core/sched/scheduler.py
git -C "$clone" merge-base --is-ancestor "$base" "$kv"
git -C "$clone" merge-base --is-ancestor "$kv" "$cap"
git -C "$clone" merge-base --is-ancestor "$cap" "$variant"

write() {  # write FILE FROM TIP DESCRIPTION: the scheduler.py part of the commits FROM..TIP
  local file=$1 from=$2 tip=$3 description=$4 sha
  {
    printf '%s\n\n' "$description"
    printf 'The scheduler.py part of these commits on vLLM %s (%s):\n\n' "$base" "$(git -C "$clone" rev-parse --short "$base^{commit}")"
    for sha in $(git -C "$clone" rev-list --reverse "$from..$tip" -- "$scheduler"); do
      picked=$(git -C "$clone" log -1 --format=%b "$sha" | sed -n 's/^(cherry picked from commit \([0-9a-f]*\))$/\1/p')
      printf '  %s %s%s\n' "$(git -C "$clone" rev-parse --short "$sha")" "$(git -C "$clone" log -1 --format=%s "$sha")" \
        "${picked:+ (cherry-picked from ${picked:0:9})}"
      git -C "$clone" log -1 --format=%b "$sha" |
        awk '/^Backport(ed to v[0-9.]+)?: /{p=1} p&&/^$/{exit} p{print "    " $0}'
    done
    printf '\n'
    git -C "$clone" diff --no-color "$from" "$tip" -- "$scheduler"
  } >"$out/$file"
}

write 63350.patch "$base" "$kv" "vLLM #65282 (https://github.com/vllm-project/vllm/pull/65282; first #63350), backported to $base."
write 63351.patch "$kv" "$cap" "vLLM #65283 (https://github.com/vllm-project/vllm/pull/65283; first #63351), backported to $base, on #65282."
write variant.patch "$cap" "$variant" "What only Flatshare's make-room variant has, on #65282 and #65283: the switches from
--additional-config's make_room table, the reserve, the retention of realtime prefixes, and the log line of the PRs'
counters and its own."
ls -l "$out"/*.patch
