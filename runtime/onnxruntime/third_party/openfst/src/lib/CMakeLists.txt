FILE(GLOB HEADER_FILES ../include/fst/*.h)

set(CMAKE_CXX_STANDARD 17)

if(WIN32)
add_library(fst STATIC
  compat.cc
  flags.cc
  fst-types.cc
  fst.cc
  mapped-file.cc
  properties.cc
  symbol-table.cc
  symbol-table-ops.cc
  util.cc
  weight.cc
  ${HEADER_FILES}
)
else()
add_library(fst
  compat.cc
  flags.cc
  fst-types.cc
  fst.cc
  mapped-file.cc
  properties.cc
  symbol-table.cc
  symbol-table-ops.cc
  util.cc
  weight.cc
  ${HEADER_FILES}
)
endif()


set_target_properties(fst PROPERTIES
  SOVERSION "${SOVERSION}"
)

find_package(glog REQUIRED)
find_package(gflags REQUIRED)

target_include_directories(fst PUBLIC ${PROJECT_SOURCE_DIR}/src/include)
target_link_libraries(fst PUBLIC glog::glog gflags::gflags)

install(TARGETS fst
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib)
