#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/fastspeech2.yaml}"
SUBSET="${2:-train-clean-100}"
WORKERS="${3:-8}"

PREPROCESSED=$(python -c "from omegaconf import OmegaConf; c=OmegaConf.load('$CONFIG'); print(c.paths.preprocessed)")

echo "Verifying feature extraction for $SUBSET..."

N_MEL=$(find "${PREPROCESSED}/${SUBSET}" -name "*_mel.npy" 2>/dev/null | wc -l)
N_PITCH=$(find "${PREPROCESSED}/${SUBSET}" -name "*_pitch.npy" 2>/dev/null | wc -l)
N_DUR=$(find "${PREPROCESSED}/${SUBSET}" -name "*_duration.npy" 2>/dev/null | wc -l)

echo "  mel spectrograms : $N_MEL"
echo "  pitch arrays     : $N_PITCH"
echo "  duration arrays  : $N_DUR"

if [[ "$N_MEL" -ne "$N_PITCH" ]] || [[ "$N_MEL" -ne "$N_DUR" ]]; then
    echo "WARNING: Feature counts do not match. Re-running preprocess for $SUBSET..."
    python -m src.preprocess \
        --config "$CONFIG" \
        --split "$SUBSET" \
        --workers "$WORKERS"
else
    echo "All features present and consistent."
fi

MANIFEST="${PREPROCESSED}/${SUBSET}_manifest.txt"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest missing. Regenerating..."
    find "${PREPROCESSED}/${SUBSET}" -name "*_mel.npy" \
        | xargs -I{} basename {} _mel.npy \
        | sort > "$MANIFEST"
    echo "Written: $MANIFEST ($(wc -l < "$MANIFEST") entries)"
else
    echo "Manifest: $MANIFEST ($(wc -l < "$MANIFEST") entries)"
fi