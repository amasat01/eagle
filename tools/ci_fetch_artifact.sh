#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Usage: ci_fetch_artifact.sh <prefix> <tree> <outdir>
# Download the newest unexpired artifact "<prefix>-<tree>" that came from a
# successful workflow run. Needs the gh CLI, GH_TOKEN and GITHUB_REPOSITORY.
# Release and deploy jobs use it so they ship what the pull request built
# and never rebuild.
set -euo pipefail
[ "$#" -eq 3 ] || { echo "usage: $0 <prefix> <tree> <outdir>" >&2; exit 2; }
prefix=$1 tree=$2 outdir=$3
name="${prefix}-${tree}"

run_ids=$(gh api "repos/${GITHUB_REPOSITORY}/actions/artifacts?name=${name}&per_page=100" \
  --jq '.artifacts | map(select(.expired | not)) | sort_by(.created_at) | reverse | .[].workflow_run.id')
for run_id in $run_ids; do
  if [ "$(gh api "repos/${GITHUB_REPOSITORY}/actions/runs/${run_id}" --jq .conclusion)" = success ]; then
    echo "artifact ${name} from run ${run_id}"
    mkdir -p "$outdir"
    gh run download "$run_id" --repo "$GITHUB_REPOSITORY" --name "$name" --dir "$outdir"
    exit 0
  fi
done
echo "::error::no successful, unexpired artifact ${name} for tree ${tree}: merge only an up-to-date PR (its checks must have run on this exact tree)"
exit 1
