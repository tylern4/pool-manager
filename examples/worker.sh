#!/usr/bin/env bash
# Minimal placeholder worker script for pool-manager testing.
# In production, this would connect back to an HTCondor startd or
# run a payload appropriate for your site.

set -euo pipefail

echo "[worker $$] starting (POOL_MANAGER_FOO=${POOL_MANAGER_FOO:-})"
sleep 60
echo "[worker $$] done"
