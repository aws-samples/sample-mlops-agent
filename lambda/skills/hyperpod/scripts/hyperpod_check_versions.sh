#!/usr/bin/env bash
# HyperPod cluster-node version audit.
#
# Invoked by the hyperpod-skill Lambda via SSM AWS-RunShellScript. Reads
# the CATEGORIES env var (comma-separated; empty = all) and emits a single
# JSON object to stdout:
#
#   {"cuda": "12.2", "nccl": "2.19.3", "pytorch": "2.4.1", ...}
#
# Unknown / unavailable tools report "not_installed". The script never
# exits non-zero on a missing tool — we want partial data over a hard
# failure, and the Lambda parses stdout as JSON regardless.
#
# Ported from awslabs/agent-plugins/plugins/sagemaker-ai/skills/hyperpod-version-checker.
set -eu
set +e  # per-probe failures are non-fatal

# Which categories did the caller ask for? Empty / unset = all.
CATS_REQUESTED="${CATEGORIES:-}"

want() {
    local cat="$1"
    if [ -z "$CATS_REQUESTED" ]; then
        return 0
    fi
    case ",$CATS_REQUESTED," in
        *,"$cat",*) return 0 ;;
        *)          return 1 ;;
    esac
}

# Each probe writes to a temp file. We assemble the final JSON at the end
# so a crashed probe can't corrupt partial output.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

emit() {
    # emit <category> <value>. Escapes backslashes + quotes for JSON.
    local cat="$1" val="$2"
    val="${val//\\/\\\\}"
    val="${val//\"/\\\"}"
    printf '"%s":"%s"' "$cat" "$val" >> "$TMP/kv"
    printf ',\n' >> "$TMP/kv"
}

# ── probes ────────────────────────────────────────────────────────────────

if want cuda; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        v=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        emit "cuda_driver" "${v:-unknown}"
    else
        emit "cuda_driver" "not_installed"
    fi
    if command -v nvcc >/dev/null 2>&1; then
        v=$(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | awk '{print $2}')
        emit "cuda_toolkit" "${v:-unknown}"
    else
        emit "cuda_toolkit" "not_installed"
    fi
fi

if want cudnn; then
    v=$(find /usr/include /usr/local/cuda -name cudnn_version.h 2>/dev/null | head -1 | xargs -r grep -E '#define CUDNN_MAJOR|#define CUDNN_MINOR|#define CUDNN_PATCHLEVEL' 2>/dev/null | awk '{print $3}' | paste -sd. -)
    emit "cudnn" "${v:-not_installed}"
fi

if want nccl; then
    v=$(find /usr /opt -name 'libnccl.so*' 2>/dev/null | grep -oE 'libnccl\.so\.[0-9.]+' | head -1 | sed 's/libnccl\.so\.//')
    emit "nccl" "${v:-not_installed}"
fi

if want efa; then
    if command -v fi_info >/dev/null 2>&1; then
        v=$(fi_info --version 2>/dev/null | head -1)
        emit "efa" "${v:-unknown}"
    else
        emit "efa" "not_installed"
    fi
fi

if want ofi-nccl; then
    v=$(find /opt -name 'libnccl-net.so*' 2>/dev/null | head -1)
    emit "ofi_nccl" "${v:-not_installed}"
fi

if want gdrcopy; then
    if command -v gdrcopy_copybw >/dev/null 2>&1; then
        emit "gdrcopy" "installed"
    else
        emit "gdrcopy" "not_installed"
    fi
fi

if want mpi; then
    if command -v mpirun >/dev/null 2>&1; then
        v=$(mpirun --version 2>/dev/null | head -1)
        emit "mpi" "${v:-unknown}"
    else
        emit "mpi" "not_installed"
    fi
fi

if want neuron; then
    if command -v neuron-ls >/dev/null 2>&1; then
        v=$(neuron-ls --version 2>/dev/null | head -1)
        emit "neuron" "${v:-unknown}"
    else
        emit "neuron" "not_installed"
    fi
fi

if want python; then
    if command -v python3 >/dev/null 2>&1; then
        v=$(python3 --version 2>&1 | awk '{print $2}')
        emit "python" "${v:-unknown}"
    else
        emit "python" "not_installed"
    fi
fi

if want pytorch; then
    v=$(python3 -c 'import torch, sys; print(torch.__version__)' 2>/dev/null)
    emit "pytorch" "${v:-not_installed}"
fi

if want runtime; then
    if command -v docker >/dev/null 2>&1; then
        v=$(docker --version 2>/dev/null | awk '{print $3}' | tr -d ',')
        emit "docker" "${v:-unknown}"
    else
        emit "docker" "not_installed"
    fi
    if command -v containerd >/dev/null 2>&1; then
        v=$(containerd --version 2>/dev/null | awk '{print $3}')
        emit "containerd" "${v:-unknown}"
    else
        emit "containerd" "not_installed"
    fi
fi

# ── assemble JSON output ──────────────────────────────────────────────────
# Strip the trailing comma from the last key=value pair.
if [ -f "$TMP/kv" ]; then
    kv=$(sed -z 's/,\n$//' "$TMP/kv")
else
    kv=""
fi
printf '{%s}\n' "$kv"
