#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath $0))
PLATFORM=$(basename $(realpath ${SCRIPT_DIR}))
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../..)
BUILD_DIR=${PROJECT_ROOT}/build

rm -rf ${BUILD_DIR}/conan_runtime
rm -rf ${BUILD_DIR}/conan_install
conan install ${PROJECT_ROOT}/conanfile.txt --deployer=runtime_deploy --deployer-folder=${BUILD_DIR}/conan_runtime -of ${BUILD_DIR}/conan_install --profile="x86_64-Linux-gcc-11.4"

mkdir -p ${BUILD_DIR}/archive
rm -rf ${BUILD_DIR}/archive/*
cp -af ${BUILD_DIR}/bin ${BUILD_DIR}/archive/bin
cp -af ${BUILD_DIR}/conan_runtime ${BUILD_DIR}/archive/runtime
