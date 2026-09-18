# PPU training integration

Status: configuration and validation tooling only; no PPU hardware results or
published FlagScale PPU image have been established yet. All changes belong on
`ppu-test-0917`. No inference/serving image is built.

## Sources and runtime contract

- Base: `harbor.baai.ac.cn/flagos-dev/flagcx:manual-20260827-ppu-dev`.
- Runner: `flagcicd-810e` (must be available to BrianPei/FlagScale).
- Device nodes and SDK paths follow [FlagCX PPU configuration](https://github.com/flagos-ai/FlagCX/blob/de5e09d62008c659c6e43d25225fc7ff207f89cd/.github/configs/ppu.yml)
  and its `run_ppu_workload.sh`: PPU exposes the `torch.cuda` API; collective
  communication uses the vendor PyTorch `nccl` backend backed by PCCL. The base
  image does not contain the optional FlagCX PyTorch plugin.
- CPU model/architecture, memory, OS, driver, firmware, SDK, torch and communication
  library versions are unknown until `inspect` runs. `ppu-smi`, `lscpu`, package
  inventory and linker cache are saved as evidence. Missing diagnostics are marked
  unavailable, not assigned guessed versions.
- Keep the vendor torch/PCCL/SDK combination. `torch_cpu` is inventoried if present,
  but is not assumed to be an independently installable dependency. Never install
  generic PyPI torch, NVIDIA Apex or CUDA FlashAttention to repair PPU imports.
- Megatron-LM-FL and TransformerEngine-FL resolve to exact SHA values from the source
  catalog. The first TE correctness baseline uses `TE_FL_SKIP_CUDA=1` and
  `TE_FL_PREFER=reference`, as supported by TE-FL. TE native CUDA `.so` is intentionally
  not built; the vendor torch/PCCL native libraries must still load and execute.
  This is not a claim of native PPU TE/FlagGems performance support.

## First hardware run

Use **PPU training image** (`ppu_train.yml`) on branch `ppu-test-0917`:

1. `operation=inspect`, `push=false`: inspect the base without installing packages.
   Download `ppu-training-<run>-<attempt>` and review missing imports and versions.
   If vendor torch/PCCL, compilers or SDK are missing, obtain the matching vendor
   packages/base image first; the build deliberately fails instead of replacing them
   with incompatible public wheels. Driver/firmware are host prerequisites.
2. Provision `/mnt/flagscale/data` and `/mnt/flagscale/tokenizers` on this runner, or
   update the two source paths in `.github/configs/ppu.yml`. Required files are listed
   in `tools/install/ppu/accept_train.sh`. These paths are proposed mount points,
   not observed existing runner directories.
3. `operation=build`: base device/autograd and two-rank NCCL/PCCL all-reduce checks,
   build with digest-pinned base and SHA-pinned sources, then candidate device,
   collective and TE BF16 forward/backward checks. Save the candidate tag from
   `candidate.txt`. This leaves the image on that runner and does not push it.
4. `operation=smoke`, `image=<candidate>`: CLI test and ten Qwen3 0.6B training
   iterations. Every iteration must produce finite positive loss. The observed loss
   artifact is **not** automatically accepted as gold. Review repeat-run stability
   and numerical correctness before committing `gold_values/0_6b_ppu.json` with
   justified tolerances. Do not copy another chip's losses or use empty gold values.
5. After committing the reviewed baseline, rebuild on the new SHA. Then run
   `operation=accept`, `image=<new candidate>`, `push=true`: post-validation, CLI,
   the full configured unit suite, ten training iterations and existing loss
   comparison must pass before push. No core dependencies are installed in tests.
   Harbor authentication must already be configured on the runner; no credentials
   belong in source or workflow logs. The pulled digest must match the tested ID.
6. Fill `ci_train_image` with the verified digest in `.github/configs/ppu.yml`.
   Run `all_tests` with `platform=ppu`. PPU is opt-in until this first promotion;
   the other platforms and shared workflows are unchanged.

The generic **Build Docker Images** workflow also discovers `image_build.tasks.train`
without changes. Before initial promotion, use `platform=ppu`, `tasks=train`,
`run_tests=false`, `push=false` for build-only debugging. Its existing cross-job
test path requires a published image and a nonempty CI consumer configuration;
use the PPU workflow above for the first test-before-push cycle.

No loss baseline or CI consumer tag is fabricated. Consequently first-time
`accept` and `all_tests/platform=ppu` deliberately cannot pass until the missing
hardware evidence has been collected. Inventory success alone is not acceptance.

Local commands on the PPU runner (Docker, GNU timeout and Mike Farah yq v4 required):

```bash
OPERATION=inspect bash tools/install/ppu/image_pipeline.sh
OPERATION=build bash tools/install/ppu/image_pipeline.sh
OPERATION=smoke CANDIDATE='<candidate tag>' bash tools/install/ppu/image_pipeline.sh
OPERATION=accept CANDIDATE='<candidate tag>' PUBLISH=true bash tools/install/ppu/image_pipeline.sh
```

Evidence is saved under `ppu-artifacts/`: image metadata, environment inventory,
source revisions, smoke/build/acceptance logs, observed losses and verified digest.
Keep raw successful and failed logs when diagnosing backend compatibility.
