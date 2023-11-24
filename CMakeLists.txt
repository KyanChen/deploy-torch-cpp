cmake_minimum_required(VERSION 3.28)
project(DeployCpp VERSION 1.0.0)
# set(CMAKE_C_FLAGS_DEBUG "-g -DDEBUG")
# set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE=Debug)

# if debug set torch dir to debug, else set torch dir to release
IF (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(Torch_DIR libs/libtorch_debug/share/cmake/Torch)
ELSE()
    set(Torch_DIR libs/libtorch/share/cmake/Torch)
ENDIF()
set(OpenCV_DIR libs/opencv/build)
set(GDAL_DIR E:/Codes/vcpkg/installed/x64-windows/share/gdal)
set(TensorRT_DIR libs/TensorRT-8.6.1.6)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GDAL CONFIG REQUIRED)

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")
message(STATUS "OpenCV library status:")
message(STATUS "    libraries: ${OpenCV_LIBRARIES}")
message(STATUS "GDAL library status:")
message(STATUS "    libraries: ${GDAL_LIBRARIES}")

include(CPack)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GDAL_INCLUDE_DIRS})
include_directories(${TensorRT_DIR}/include)

link_directories(${TensorRT_DIR}/lib)

#add_executable(DeployCpp ctest.cpp)
add_executable(DeployCpp main_multithread_trt.cpp)

target_link_libraries(DeployCpp ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES} ${GDAL_LIBRARIES})
target_link_libraries(DeployCpp nvinfer)
target_link_libraries(DeployCpp nvinfer_plugin)
target_link_libraries(DeployCpp nvparsers)
target_link_libraries(DeployCpp nvonnxparser)
set_property(TARGET DeployCpp PROPERTY CXX_STANDARD 17)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET DeployCpp
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:DeployCpp>)
    file(GLOB OpenCV_DLLS "${OpenCV_DIR}/x64/vc16/bin/*.dll")
    add_custom_command(TARGET DeployCpp
          POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${OpenCV_DLLS}
          $<TARGET_FILE_DIR:DeployCpp>)
    file(GLOB TensorRT_DLLS "${TensorRT_DIR}/lib/*.dll")
    add_custom_command(TARGET DeployCpp
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TensorRT_DLLS}
            $<TARGET_FILE_DIR:DeployCpp>)

endif (MSVC)
