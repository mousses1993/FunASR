#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath $0))
PLATFORM=$(basename $(realpath ${SCRIPT_DIR}))
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../..)
BUILD_DIR=${PROJECT_ROOT}/build

cmake   -DPLATFORM=${PLATFORM} \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=${PROJECT_ROOT}/cmake/conan_provider.cmake \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DGPU=OFF \
        -DONNXRUNTIME_USE_CUDA=ON \
        -B ${PROJECT_ROOT}/build \
        -S ${PROJECT_ROOT} \
        -G Ninja 


cmake --build ${BUILD_DIR}
