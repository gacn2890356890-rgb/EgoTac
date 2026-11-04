#!/usr/bin/env bash
# Prove EgoFound3R_original is byte-identical to the official release at commit 264f4df.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../EgoFound3R_original"
if command -v sha256sum >/dev/null 2>&1; then CHECK="sha256sum --check --quiet"; else CHECK="shasum -a 256 --check --quiet"; fi
$CHECK ../scripts/original_checksums.sha256
extra=$(find . -type f -not -name '*.pyc' | wc -l); expected=$(wc -l < ../scripts/original_checksums.sha256)
[ "$extra" -eq "$expected" ] || { echo "error: $extra files in EgoFound3R_original, expected $expected" >&2; exit 1; }
echo "EgoFound3R_original matches the official release ($expected files)"
