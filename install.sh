#!/bin/bash
# install.sh — swap setup.cfg and pip install -e .
#
# Usage:
#   ./install.sh        → full mrfsim environment (TF + VoxelMorph)
#   ./install.sh hosvd  → HOSVD-only environment (no TF, no VoxelMorph)

set -e  # exit on any error

FULL_CFG="setup.cfg"
HOSVD_CFG="setup_hosvd.cfg"
BACKUP_CFG="setup.cfg.bak"

if [ "$1" = "hosvd" ]; then
    echo "==> Installing HOSVD-only environment (no TF, no VoxelMorph)..."

    if [ ! -f "$HOSVD_CFG" ]; then
        echo "ERROR: $HOSVD_CFG not found." >&2
        exit 1
    fi

    cp "$FULL_CFG" "$BACKUP_CFG"
    cp "$HOSVD_CFG" "$FULL_CFG"

    pip install -e .
    EXIT_CODE=$?

    mv "$BACKUP_CFG" "$FULL_CFG"

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: pip install failed. $FULL_CFG has been restored." >&2
        exit $EXIT_CODE
    fi

    echo "==> Done. Active environment: HOSVD-only."

else
    echo "==> Installing full mrfsim environment (TF + VoxelMorph)..."

    if [ ! -f "$FULL_CFG" ]; then
        echo "ERROR: $FULL_CFG not found." >&2
        exit 1
    fi

    pip install -e .

    echo "==> Done. Active environment: full mrfsim."
fi
