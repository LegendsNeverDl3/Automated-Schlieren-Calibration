#!/usr/bin/env bash

# Determine script root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR"

# Version parameter (defaults to 1 if not specified)
VERSION="${1:-1}"

# Normalize version argument
case "$VERSION" in
    1|v1|V1)
        VER_ARG="1"
        VER_DESC="Version 1 (Raw Mean + Variance Baseline, Recommended)"
        ;;
    2|v2|V2)
        VER_ARG="2"
        VER_DESC="Version 2 (CLAHE Contrast Normalized)"
        ;;
    -h|--help|help)
        echo "Usage: ./run_ml.sh [VERSION] [ADDITIONAL_OPTIONS]"
        echo ""
        echo "Arguments:"
        echo "  VERSION              Model version to run: 1 or 2 (default: 1)"
        echo ""
        echo "Examples:"
        echo "  ./run_ml.sh          # Runs default Version 1"
        echo "  ./run_ml.sh 1        # Runs Version 1"
        echo "  ./run_ml.sh 2        # Runs Version 2"
        echo "  ./run_ml.sh 1 --cam 0"
        echo ""
        exit 0
        ;;
    *)
        echo "[Warning] Unknown version '$VERSION'. Valid options are 1 or 2."
        echo "Defaulting to Version 1."
        VER_ARG="1"
        VER_DESC="Version 1 (Raw Mean + Variance Baseline, Recommended)"
        ;;
esac

# Shift first argument if it was a version specifier so remaining flags pass to python
if [ "$#" -ge 1 ]; then
    shift
fi

echo "=========================================================="
echo " Starting ML Auto-Schlieren"
echo " Active Model: $VER_DESC"
echo " Executing   : python schlieren_ml/ml_auto_schlieren.py --version $VER_ARG $@"
echo "=========================================================="

python schlieren_ml/ml_auto_schlieren.py --version "$VER_ARG" "$@"
