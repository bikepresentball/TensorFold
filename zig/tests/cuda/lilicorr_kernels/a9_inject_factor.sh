#!/bin/bash
# A9 fail-before injector - FACTOR control: the cross-term W-slab fold at the PRODUCTION entry.

# Re-introduces the slab mis-index class against the scratch copy of the kernel header's

# tf_lilicorr_factor_rows (the entry cuda_lilicorr_round.zig:315 launches): the cross term

# reads the anchor-term slab (w[hd + ii]) instead of its own (w[2 * hd + ii]). The device

# factor output then mis-folds every row at the production shape (rows 56, hd 1024, grid 128)

# and the FACTOR control's host mirror grades RED. The TAIL control keeps its goldens: its

# w2 slab is zero AND its anchor is all zero, so the folded term contributes exactly zero.

# Never edits the repo tree: the caller runs this against a COPY (see a9_run.sh).

# Usage: bash a9_inject_factor.sh <scratch-dir>

set -eu

D="${1:?usage: bash a9_inject_factor.sh <scratch-dir>}"
F="$D/qwen35_draft_lilicorr.cuh"
grep -q "A9-INJECTOR-FACTOR" "$F" && { echo "already injected"; exit 1; }
grep -q "const float w3 = ok ? w\[2 \* hd + ii\] : 0.f;" "$F" || { echo "cure line not found in $F - refusing"; exit 1; }
sed -i 's|const float w3 = ok ? w\[2 \* hd + ii\] : 0.f;|const float w3 = ok ? w[hd + ii] : 0.f;  // A9-INJECTOR-FACTOR: slab fold|' "$F"
grep -q "A9-INJECTOR-FACTOR" "$F" || { echo "sed did not apply"; exit 1; }
echo "A9-INJECTOR-FACTOR: applied (cross term folded onto the w2 slab, production factor_rows)"
