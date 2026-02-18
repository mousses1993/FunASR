cmake_minimum_required(VERSION 3.25)

include(${CMAKE_CURRENT_SOURCE_DIR}/recipes/${PLATFORM}/platform.cmake)

project(FunASROnnx)

option(ENABLE_FST "Whether to build openfst" ON) # ITN need openfst compiled
option(GPU "Whether to build with GPU" OFF)

# set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(TestBigEndian)
test_big_endian(BIG_ENDIAN)
if(BIG_ENDIAN)
    message("Big endian system")
else()
    message("Little endian system")
endif()

find_package(onnxruntime REQUIRED)
find_package(ffmpeg REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(glog REQUIRED)
find_package(gflags REQUIRED)

# for onnxruntime
if(ONNXRUNTIME_USE_CUDA)
    add_definitions(-DONNXRUNTIME_USE_CUDA)
endif()

IF(WIN32)
    # file(REMOVE ${PROJECT_SOURCE_DIR}/third_party/glog/src/config.h 
    #             ${PROJECT_SOURCE_DIR}/third_party/glog/src/glog/export.h 
    #             ${PROJECT_SOURCE_DIR}/third_party/glog/src/glog/logging.h 
    #             ${PROJECT_SOURCE_DIR}/third_party/glog/src/glog/raw_logging.h 
    #             ${PROJECT_SOURCE_DIR}/third_party/glog/src/glog/stl_logging.h 
    #             ${PROJECT_SOURCE_DIR}/third_party/glog/src/glog/vlog_is_on.h)
endif()
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/third_party/kaldi-native-fbank)
include_directories(${PROJECT_SOURCE_DIR}/third_party/yaml-cpp/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/jieba/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/jieba/include/limonp/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/kaldi)


if(GPU)
    add_definitions(-DUSE_GPU)
    set(TORCH_DIR "/usr/local/lib/python3.8/dist-packages/torch")
    set(TORCH_BLADE_DIR "/usr/local/lib/python3.8/dist-packages/torch_blade")
    include_directories(${TORCH_DIR}/include)
    include_directories(${TORCH_DIR}/include/torch/csrc/api/include)
    link_directories(${TORCH_DIR}/lib)
    link_directories(${TORCH_BLADE_DIR})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -D_GLIBCXX_USE_CXX11_ABI=0")
endif()


if(ENABLE_FST)
    # the following openfst if cloned from https://github.com/kkm000/openfst.git
    # with some patch to fix the make errors. 
    add_subdirectory(third_party/openfst)
    include_directories(${openfst_SOURCE_DIR}/src/include)
    if(WIN32)
    include_directories(${openfst_SOURCE_DIR}/src/lib)
    endif()
endif()

add_subdirectory(third_party/yaml-cpp)
add_subdirectory(third_party/kaldi-native-fbank/kaldi-native-fbank/csrc)
add_subdirectory(third_party/kaldi)
add_subdirectory(src)
add_subdirectory(bin)
