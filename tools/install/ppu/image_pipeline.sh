#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
set -euo pipefail
config=.github/configs/ppu.yml
operation=${OPERATION:-inspect}
publish=${PUBLISH:-false}
[[ "$publish" = false || "$operation" = accept ]] || {
    echo 'Publishing requires operation=accept.' >&2; exit 2;
}
mkdir -p ppu-artifacts
export IMAGE_BUILD_TASK=train
export IMAGE_BUILD_BASE_IMAGE
IMAGE_BUILD_BASE_IMAGE=$(yq -r '.image_build.tasks.train.base_image' "$config")
case "$operation" in
    inspect)
        # Inventory deliberately works even if PyTorch/FlagCX is absent.
        docker pull "$IMAGE_BUILD_BASE_IMAGE"
        docker image inspect "$IMAGE_BUILD_BASE_IMAGE" > ppu-artifacts/base-image.json
        bash tools/install/ppu/run_container.sh "$IMAGE_BUILD_BASE_IMAGE" \
            python tools/install/ppu/probe.py inventory | tee ppu-artifacts/base-inventory.json
        ;;
    build)
        export IMAGE_BUILD_PHASE=pre
        bash tools/install/ppu/validate_image_build.sh
        source_sha=$(git rev-parse HEAD)
        candidate="harbor.baai.ac.cn/flagos-dev/flagscale-train:${source_sha:0:12}-ppu-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-dev"
        build_args=()
        for pair in FLAGSCALE_MEGATRON_REF:megatron_lm_fl FLAGSCALE_TE_REF:transformer_engine_fl; do
            arg=${pair%%:*}; key=${pair#*:}
            repository=$(KEY="$key" yq -r '.sources[strenv(KEY)].repository' .github/configs/image_sources.yml)
            branch=$(KEY="$key" yq -r '.sources[strenv(KEY)].branch' .github/configs/image_sources.yml)
            revision=$(git ls-remote "$repository" "refs/heads/$branch" | cut -f1)
            [[ "$revision" =~ ^[0-9a-f]{40}$ ]]
            build_args+=(--build-arg "$arg=$revision")
            printf '%s %s %s\n' "$key" "$revision" "$repository" >> ppu-artifacts/sources.txt
        done
        # Pin the actual base pulled by pre-validation, not its mutable tag.
        base_digest=$(docker image inspect "$IMAGE_BUILD_BASE_IMAGE" --format '{{index .RepoDigests 0}}')
        docker build --file docker/ppu/Dockerfile.train --target dev \
            --build-arg "BASE_IMAGE=$base_digest" --build-arg "FLAGSCALE_REVISION=$source_sha" \
            "${build_args[@]}" --tag "$candidate" . 2>&1 | tee ppu-artifacts/build.log
        export IMAGE_BUILD_PHASE=post IMAGE_BUILD_CANDIDATE_IMAGE="$candidate"
        bash tools/install/ppu/validate_image_build.sh
        printf '%s\n' "$candidate" | tee ppu-artifacts/candidate.txt
        ;;
    smoke|accept)
        candidate=${CANDIDATE:?Specify the locally built candidate tag}
        [[ "$candidate" =~ ^harbor\.baai\.ac\.cn/flagos-dev/flagscale-train:[a-zA-Z0-9_.-]+$ ]] || exit 2
        # Do not rebuild or pull a different image between acceptance and push.
        image_id=$(docker image inspect "$candidate" --format '{{.Id}}')
        revision=$(docker image inspect "$candidate" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')
        test "$revision" = "$(git rev-parse HEAD)" || {
            echo 'Candidate source revision differs from checked-out branch; rebuild.' >&2; exit 1;
        }
        export IMAGE_BUILD_PHASE=post IMAGE_BUILD_CANDIDATE_IMAGE="$image_id"
        bash tools/install/ppu/validate_image_build.sh
        PPU_WITH_DATA=1 timeout 5400 bash tools/install/ppu/run_container.sh "$image_id" \
            bash tools/install/ppu/accept_train.sh "$operation" 2>&1 | tee ppu-artifacts/acceptance.log
        if [ "$publish" = true ]; then
            test "$(docker image inspect "$candidate" --format '{{.Id}}')" = "$image_id"
            # Runner must already have Harbor credentials configured by its operator.
            docker push "$candidate" 2>&1 | tee ppu-artifacts/push.log
            digest=$(docker image inspect "$candidate" --format '{{index .RepoDigests 0}}')
            [[ "$digest" == harbor.baai.ac.cn/flagos-dev/flagscale-train@sha256:* ]]
            docker pull "$digest"
            test "$(docker image inspect "$digest" --format '{{.Id}}')" = "$image_id"
            printf '%s\n' "$digest" | tee ppu-artifacts/verified-digest.txt
        fi
        ;;
    *) echo "Unknown operation: $operation" >&2; exit 2 ;;
esac
