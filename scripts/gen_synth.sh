#!/bin/bash
# usage: gen_synth.sh OUT_DIR COUNT TAG SEED
set -e
OUT=$1; N=$2; TAG=$3; SEED=$4
gen() {
    uv run python -m dataset.ctt_generator --count $N --size $1 --seed $3 --jobs $4 --solver-workers $5 \
        --out-dir $OUT --prefix $2 > outputs/gen_${TAG}_$1.log 2>&1
    ls $OUT/$2[0-9]*.ctt | xargs -P 16 -n 200 uv run python -m dataset.ctt_encode > outputs/enc_${TAG}_$1.log 2>&1
}
gen small  gen$TAG $SEED             8  2 &
gen medium mid$TAG $((SEED + 1000000)) 8  4 &
gen large  lar$TAG $((SEED + 2000000)) 12 4 &
wait
