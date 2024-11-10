set(TARGET kernels)

file(GLOB kernels_srcs CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/kernels/*.cu")
message(STATUS "Kernels: ${kernels_srcs}")

add_library(${TARGET} OBJECT ${kernels_srcs})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/include/tools)
target_compile_options(${TARGET} PRIVATE "-lineinfo")
set_target_properties(${TARGET} PROPERTIES CUDA_PTX_COMPILATION ON)

# The only way I found to get paths to OBJ/PTX files is to use generator expressions which can be used only in certain contexts.
add_custom_target(move_ptx ALL
  # Not sure why replacing ';' with '\n' works differently than changing delimiter to ';' in xargs, but for some reson it is needed here
  COMMAND echo "$<TARGET_OBJECTS:${TARGET}>" | sed "s/\;/\\n/g" | xargs -I {} sh -c "cp {} ${CMAKE_CURRENT_BINARY_DIR}/$(basename {})" 
  VERBATIM
  DEPENDS ${TARGET}
)
