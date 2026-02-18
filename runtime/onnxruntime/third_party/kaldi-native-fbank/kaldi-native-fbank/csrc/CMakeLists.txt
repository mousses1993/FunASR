add_library(csrc STATIC
        feature-fbank.cc
        feature-functions.cc
        feature-window.cc
        fftsg.c
        mel-computations.cc
        online-feature.cc
        rfft.cc)

target_include_directories(csrc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)