# Install script for directory: /home/main/lihy/RTG-SLAM/thirdParty/pybind

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/main/anaconda3/envs/RTG-SLAM/lib/python3.9/site-packages")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/main/anaconda3/envs/RTG-SLAM/lib/python3.9/site-packages/orbslam2.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/main/anaconda3/envs/RTG-SLAM/lib/python3.9/site-packages/orbslam2.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/main/anaconda3/envs/RTG-SLAM/lib/python3.9/site-packages/orbslam2.so"
         RPATH "/home/main/lihy/RTG-SLAM/thirdParty/install/lib:/home/main/anaconda3/envs/RTG-SLAM/lib")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/main/anaconda3/envs/RTG-SLAM/lib/python3.9/site-packages/orbslam2.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/main/anaconda3/envs/RTG-SLAM/lib/python3.9/site-packages" TYPE SHARED_LIBRARY FILES "/home/main/lihy/RTG-SLAM/thirdParty/pybind/lib/orbslam2.so")
  if(EXISTS "$ENV{DESTDIR}/home/main/anaconda3/envs/RTG-SLAM/lib/python3.9/site-packages/orbslam2.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/main/anaconda3/envs/RTG-SLAM/lib/python3.9/site-packages/orbslam2.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/main/anaconda3/envs/RTG-SLAM/lib/python3.9/site-packages/orbslam2.so"
         OLD_RPATH "/home/main/lihy/RTG-SLAM/thirdParty/pybind/build/../../install/lib:/home/main/lihy/RTG-SLAM/thirdParty/install/lib:/home/main/anaconda3/envs/RTG-SLAM/lib:"
         NEW_RPATH "/home/main/lihy/RTG-SLAM/thirdParty/install/lib:/home/main/anaconda3/envs/RTG-SLAM/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/main/anaconda3/envs/RTG-SLAM/lib/python3.9/site-packages/orbslam2.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/main/lihy/RTG-SLAM/thirdParty/pybind/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
