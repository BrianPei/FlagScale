#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
set -euo pipefail
mode=${1:-accept}
[[ "$mode" = accept || "$mode" = smoke ]] || exit 2
source tools/install/ppu/env.sh
for file in \
    /home/gitlab-runner/data/pile_wikipedia_demo/pile_wikipedia_demo.bin \
    /home/gitlab-runner/data/pile_wikipedia_demo/pile_wikipedia_demo.idx \
    /home/gitlab-runner/tokenizers/qwentokenizer/qwen.tiktoken \
    /home/gitlab-runner/tokenizers/qwentokenizer/tokenizer_config.json \
    /home/gitlab-runner/tokenizers/qwentokenizer/tokenization_qwen.py; do
    test -r "$file" || { echo "Missing training asset: $file" >&2; exit 1; }
done
flagscale --version
python -m pytest tests/unit_tests/test_cli.py -v
if [ "$mode" = accept ]; then
    gold=tests/functional_tests/train/qwen3/gold_values/0_6b_ppu.json
    test -s "$gold" || {
        echo "PPU loss baseline missing: run smoke, review the recorded loss, then commit $gold." >&2
        exit 1
    }
    bash tests/test_utils/runners/run_tests.sh --platform ppu --device ppu --type unit
fi
# --test waits for training completion. Keep the general result comparator intact.
rm -rf tests/functional_tests/train/qwen3/test_results/0_6b_ppu
flagscale train qwen3 --config \
    tests/functional_tests/train/qwen3/conf/0_6b_ppu.yaml --test
python tools/install/ppu/check_train_log.py
if [ "$mode" = accept ]; then
    python -m pytest tests/test_utils/runners/check_results.py::test_train_equal \
        --path=tests/functional_tests --task=train --model=qwen3 \
        --case=0_6b_ppu --platform=ppu --device=ppu
fi
