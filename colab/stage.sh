#!/bin/bash
# Usage on the VM: nohup bash colab/job.sh STAGE bash colab/stage.sh STAGE [ARGS...] &
# Stages: setup, prep, ab (A/B of the two unit kinds), train NAME UNITS [train.py flags...], test RUN.
set -euo pipefail
D=/content/data
R=/content/runs
case $1 in
  setup)  # Colab has torch already. Do not reinstall it.
    nproc; free -g | head -2; df -h /content | tail -1; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    grep -v '^torch' requirements.txt > /tmp/req.txt
    uv pip install --system -q -r /tmp/req.txt
    python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
    hf download google/embeddinggemma-2 > /dev/null
    python tests/test_data.py
    python tests/test_model.py ;;
  prep)
    python scripts/prepare.py --sources prajwal mile fleurs --raw_dir $D/raw --out_dir $D/prepared --workers "$(nproc)"
    df -h /content | tail -1 ;;
  ab)  # same budget for both unit kinds; the lower average valid WER wins
    for u in gemma bpe; do
      python scripts/train.py --data_dir $D/prepared --units $D/prepared/units_$u.json --out_dir $R/ab_$u \
        --reader base --max_steps 2500 --warmup_steps 500 --max_minutes 35 --eval_every 1250 --save_every 1250 \
        --num_workers "$(( $(nproc) - 2 ))"
    done ;;
  train)  # train NAME UNITS [more train.py flags]
    name=$2 units=$3; shift 3
    python scripts/train.py --data_dir $D/prepared --units $D/prepared/units_$units.json --out_dir $R/$name \
      --num_workers "$(( $(nproc) - 2 ))" "$@" ;;
  test)  # test RUN: every test set, no cap
    python scripts/train.py --data_dir $D/prepared --eval_only --resume $R/$2/best.pt --eval_split test --out_dir $R/$2 ;;
  *) echo "unknown stage $1"; exit 2 ;;
esac
