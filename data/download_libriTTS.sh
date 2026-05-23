#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-/projects/e32706/speech_synthesis/data/LibriTTS}"
N_JOBS="${2:-8}"

mkdir -p "$DEST"

declare -a SUBSETS=(
    "train-clean-100"
    "train-clean-360"
    "dev-clean"
    "test-clean"
)

BASE_URL="https://www.openslr.org/resources/60"

download_subset() {
    local subset="$1"
    local dest="$2"
    local url="${BASE_URL}/${subset}.tar.gz"
    local archive="${dest}/${subset}.tar.gz"

    if [[ -d "${dest}/${subset}" ]]; then
        echo "[SKIP] ${subset} already extracted"
        return 0
    fi

    echo "[DL] ${subset}"
    wget -q --show-progress -c "$url" -O "$archive"
    echo "[EXTRACT] ${subset}"
    tar -xzf "$archive" -C "$dest" --strip-components=1
    rm -f "$archive"
    echo "[DONE] ${subset}"
}

export -f download_subset
export BASE_URL

parallel -j "$N_JOBS" download_subset {} "$DEST" ::: "${SUBSETS[@]}"

echo "All subsets downloaded to $DEST"