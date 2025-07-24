cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM2_PYTHON_BINDING)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++17 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX17)
   message("CXX17")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
   add_definitions(-DCOMPILEDWITHC17)
   message(STATUS "Using flag -std=c++17.")
elseif(COMPILER_SUPPORTS_CXX0X)
   message("CXX14")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

find_package(NumPy)
find_package(OpenCV 4.2)

find_package(Eigen3 3.1.0 REQUIRED)
# RPATH fiddling for the library, so that the installed version can find the linked libraries
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories(
${PROJECT_SOURCE_DIR}/src
${ORB_SLAM2_INCLUDE_DIR}
${Boost_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR}
${PYTHON_INCLUDE_DIRS}
${PYTHON_NUMPY_INCLUDE_DIR}
)

find_package(Pangolin REQUIRED)

if (Pangolin_FOUND)
  include_directories(${Pangolin_INCLUDE_DIRS})
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(TARGET_MODULE_NAME orbslam2)
add_library(${TARGET_MODULE_NAME} SHARED
src/ORBSlamPython.cpp
src/pyboost_cv2_converter.cpp
src/pyboost_cv3_converter.cpp)
set_target_properties(${TARGET_MODULE_NAME} PROPERTIES PREFIX "")

message(${ORB_SLAM2_LIBRARIES})

target_link_libraries(${TARGET_MODULE_NAME} 
   ${ORB_SLAM2_LIBRARIES}
   ${OpenCV_LIBS}
   ${EIGEN3_LIBS}
   ${Boost_LIBRARIES}
   ${PYTHON_LIBRARIES}
)

install(TARGETS ${TARGET_MODULE_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX})

