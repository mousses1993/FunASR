
file(GLOB files1 "*.cpp")
if(APPLE)
    file(GLOB itn_files "itn-*.cpp")
    list(REMOVE_ITEM files1 ${itn_files})
endif(APPLE)
list(REMOVE_ITEM files1 "${CMAKE_CURRENT_SOURCE_DIR}/paraformer-torch.cpp")
set(files ${files1})

if(GPU)
    set(files ${files} "${CMAKE_CURRENT_SOURCE_DIR}/paraformer-torch.cpp")
endif()

message("files: "${files})

if(WIN32)
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/execution-charset:utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/bigobj>")
endif()

find_package(yaml-cpp REQUIRED)

add_library(funasr SHARED ${files})

if(WIN32)
    set(EXTRA_LIBS yaml-cpp::yaml-cpp csrc kaldi-decoder fst glog gflags ffmpeg::ffmpeg)
    # include_directories(${onnxruntime_INCLUDE_DIRS})
    # include_directories(${ffmpeg_INCLUDE_DIRS})
    # target_link_directories(funasr PUBLIC ${onnxruntime_LIBRARY_DIRS})
    # target_link_directories(funasr PUBLIC ${ffmpeg_LIBRARY_DIRS})
    target_compile_definitions(funasr PUBLIC -D_FUNASR_API_EXPORT -DNOMINMAX -DYAML_CPP_DLL)
else()
    set(EXTRA_LIBS pthread yaml-cpp::yaml-cpp csrc kaldi-decoder fst glog::glog gflags::gflags nlohmann_json::nlohmann_json ONNXRuntime::onnxruntime ffmpeg::ffmpeg)
    # include_directories(${onnxruntime_INCLUDE_DIRS})
    # include_directories(${ffmpeg_INCLUDE_DIRS})
    # if(APPLE)
    #     target_link_directories(funasr PUBLIC ${onnxruntime_LIBRARY_DIRS})
    #     target_link_directories(funasr PUBLIC ${ffmpeg_LIBRARY_DIRS})
    # endif(APPLE)    
endif()

if(GPU)
    set(TORCH_DEPS torch torch_cuda torch_cpu c10 c10_cuda torch_blade ral_base_context)
endif()

if(ONNXRUNTIME_USE_CUDA)
    list(APPEND EXTRA_LIBS cuDNN::cuDNN CUDA::cudart)
    target_compile_definitions(funasr PUBLIC -DONNXRUNTIME_USE_CUDA)
endif()


#message("CXX_FLAGS "${CMAKE_CXX_FLAGS})
target_include_directories(funasr PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(funasr PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(funasr PUBLIC ${PROJECT_SOURCE_DIR}/third_party)
target_include_directories(funasr PUBLIC ${PROJECT_SOURCE_DIR}/third_party/jieba/include)

target_link_libraries(funasr PUBLIC ${EXTRA_LIBS} ${TORCH_DEPS})
