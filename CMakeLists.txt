cmake_minimum_required(VERSION 3.4)
project(MultiScale)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS "-fopenmp")
IF (CMAKE_BUILD_TYPE MATCHES "Debug")
        MESSAGE(STATUS "CMAKE_BUILD_TYPE = Debug")
ELSEIF (CMAKE_BUILD_TYPE MATCHES "Release")
        MESSAGE(STATUS "CMAKE_BUILD_TYPE = Release")
ELSE()
        set(CMAKE_BUILD_TYPE "Release")
        MESSAGE(STATUS "Set defalult CMAKE_BUILD_TYPE = Release")
ENDIF()

SET(LIBRARIES
        opencv_xfeatures2d
        opencv_calib3d
        opencv_features2d
        opencv_flann
        opencv_highgui
        opencv_imgcodecs
        opencv_imgproc
        opencv_core
        )

find_package(CUDA)
if(CUDA_FOUND)
        set(LIBRARIES "${LIBRARIES}" 
                /usr/local/lib/colmap/libsift_gpu.a
                /usr/local/cuda-10.0/lib64/libcudart.so
                OpenGL
                glut
                GLEW)
else()
	MESSAGE("-- Can't find CUDA. Continuing without it.")
endif()

add_executable(MultiScale main.cpp)
target_link_libraries(MultiScale ${LIBRARIES})
