#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
FFMPEG_DIR=${SCRIPT_DIR}/third_party/ffmpeg-N-111383-g20b8688092-linux64-gpl-shared
ONNXRUNTIME_DIR=${SCRIPT_DIR}/third_party/onnxruntime-linux-x64-gpu-1.24.1

cmake \
    -DFFMPEG_DIR=${FFMPEG_DIR} \
    -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -S ${SCRIPT_DIR} \
    -B ${SCRIPT_DIR}/build

cmake --build ${SCRIPT_DIR}/build -j10