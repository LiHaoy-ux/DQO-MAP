# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/main/lihy/RTG-SLAM/thirdParty/pybind

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/main/lihy/RTG-SLAM/thirdParty/pybind/build

# Include any dependencies generated for this target.
include CMakeFiles/orbslam2.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/orbslam2.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/orbslam2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/orbslam2.dir/flags.make

CMakeFiles/orbslam2.dir/src/ORBSlamPython.cpp.o: CMakeFiles/orbslam2.dir/flags.make
CMakeFiles/orbslam2.dir/src/ORBSlamPython.cpp.o: ../src/ORBSlamPython.cpp
CMakeFiles/orbslam2.dir/src/ORBSlamPython.cpp.o: CMakeFiles/orbslam2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/main/lihy/RTG-SLAM/thirdParty/pybind/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/orbslam2.dir/src/ORBSlamPython.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/orbslam2.dir/src/ORBSlamPython.cpp.o -MF CMakeFiles/orbslam2.dir/src/ORBSlamPython.cpp.o.d -o CMakeFiles/orbslam2.dir/src/ORBSlamPython.cpp.o -c /home/main/lihy/RTG-SLAM/thirdParty/pybind/src/ORBSlamPython.cpp

CMakeFiles/orbslam2.dir/src/ORBSlamPython.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orbslam2.dir/src/ORBSlamPython.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/main/lihy/RTG-SLAM/thirdParty/pybind/src/ORBSlamPython.cpp > CMakeFiles/orbslam2.dir/src/ORBSlamPython.cpp.i

CMakeFiles/orbslam2.dir/src/ORBSlamPython.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orbslam2.dir/src/ORBSlamPython.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/main/lihy/RTG-SLAM/thirdParty/pybind/src/ORBSlamPython.cpp -o CMakeFiles/orbslam2.dir/src/ORBSlamPython.cpp.s

CMakeFiles/orbslam2.dir/src/pyboost_cv2_converter.cpp.o: CMakeFiles/orbslam2.dir/flags.make
CMakeFiles/orbslam2.dir/src/pyboost_cv2_converter.cpp.o: ../src/pyboost_cv2_converter.cpp
CMakeFiles/orbslam2.dir/src/pyboost_cv2_converter.cpp.o: CMakeFiles/orbslam2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/main/lihy/RTG-SLAM/thirdParty/pybind/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/orbslam2.dir/src/pyboost_cv2_converter.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/orbslam2.dir/src/pyboost_cv2_converter.cpp.o -MF CMakeFiles/orbslam2.dir/src/pyboost_cv2_converter.cpp.o.d -o CMakeFiles/orbslam2.dir/src/pyboost_cv2_converter.cpp.o -c /home/main/lihy/RTG-SLAM/thirdParty/pybind/src/pyboost_cv2_converter.cpp

CMakeFiles/orbslam2.dir/src/pyboost_cv2_converter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orbslam2.dir/src/pyboost_cv2_converter.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/main/lihy/RTG-SLAM/thirdParty/pybind/src/pyboost_cv2_converter.cpp > CMakeFiles/orbslam2.dir/src/pyboost_cv2_converter.cpp.i

CMakeFiles/orbslam2.dir/src/pyboost_cv2_converter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orbslam2.dir/src/pyboost_cv2_converter.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/main/lihy/RTG-SLAM/thirdParty/pybind/src/pyboost_cv2_converter.cpp -o CMakeFiles/orbslam2.dir/src/pyboost_cv2_converter.cpp.s

CMakeFiles/orbslam2.dir/src/pyboost_cv3_converter.cpp.o: CMakeFiles/orbslam2.dir/flags.make
CMakeFiles/orbslam2.dir/src/pyboost_cv3_converter.cpp.o: ../src/pyboost_cv3_converter.cpp
CMakeFiles/orbslam2.dir/src/pyboost_cv3_converter.cpp.o: CMakeFiles/orbslam2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/main/lihy/RTG-SLAM/thirdParty/pybind/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/orbslam2.dir/src/pyboost_cv3_converter.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/orbslam2.dir/src/pyboost_cv3_converter.cpp.o -MF CMakeFiles/orbslam2.dir/src/pyboost_cv3_converter.cpp.o.d -o CMakeFiles/orbslam2.dir/src/pyboost_cv3_converter.cpp.o -c /home/main/lihy/RTG-SLAM/thirdParty/pybind/src/pyboost_cv3_converter.cpp

CMakeFiles/orbslam2.dir/src/pyboost_cv3_converter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orbslam2.dir/src/pyboost_cv3_converter.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/main/lihy/RTG-SLAM/thirdParty/pybind/src/pyboost_cv3_converter.cpp > CMakeFiles/orbslam2.dir/src/pyboost_cv3_converter.cpp.i

CMakeFiles/orbslam2.dir/src/pyboost_cv3_converter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orbslam2.dir/src/pyboost_cv3_converter.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/main/lihy/RTG-SLAM/thirdParty/pybind/src/pyboost_cv3_converter.cpp -o CMakeFiles/orbslam2.dir/src/pyboost_cv3_converter.cpp.s

# Object files for target orbslam2
orbslam2_OBJECTS = \
"CMakeFiles/orbslam2.dir/src/ORBSlamPython.cpp.o" \
"CMakeFiles/orbslam2.dir/src/pyboost_cv2_converter.cpp.o" \
"CMakeFiles/orbslam2.dir/src/pyboost_cv3_converter.cpp.o"

# External object files for target orbslam2
orbslam2_EXTERNAL_OBJECTS =

../lib/orbslam2.so: CMakeFiles/orbslam2.dir/src/ORBSlamPython.cpp.o
../lib/orbslam2.so: CMakeFiles/orbslam2.dir/src/pyboost_cv2_converter.cpp.o
../lib/orbslam2.so: CMakeFiles/orbslam2.dir/src/pyboost_cv3_converter.cpp.o
../lib/orbslam2.so: CMakeFiles/orbslam2.dir/build.make
../lib/orbslam2.so: ../../install/lib/libORB_SLAM2.so
../lib/orbslam2.so: /home/main/lihy/RTG-SLAM/thirdParty/install/lib/libopencv_dnn.so.4.2.0
../lib/orbslam2.so: /home/main/lihy/RTG-SLAM/thirdParty/install/lib/libopencv_gapi.so.4.2.0
../lib/orbslam2.so: /home/main/lihy/RTG-SLAM/thirdParty/install/lib/libopencv_highgui.so.4.2.0
../lib/orbslam2.so: /home/main/lihy/RTG-SLAM/thirdParty/install/lib/libopencv_ml.so.4.2.0
../lib/orbslam2.so: /home/main/lihy/RTG-SLAM/thirdParty/install/lib/libopencv_objdetect.so.4.2.0
../lib/orbslam2.so: /home/main/lihy/RTG-SLAM/thirdParty/install/lib/libopencv_photo.so.4.2.0
../lib/orbslam2.so: /home/main/lihy/RTG-SLAM/thirdParty/install/lib/libopencv_stitching.so.4.2.0
../lib/orbslam2.so: /home/main/lihy/RTG-SLAM/thirdParty/install/lib/libopencv_video.so.4.2.0
../lib/orbslam2.so: /home/main/lihy/RTG-SLAM/thirdParty/install/lib/libopencv_videoio.so.4.2.0
../lib/orbslam2.so: ../../install/lib/libboost_python39.so
../lib/orbslam2.so: /home/main/anaconda3/envs/RTG-SLAM/lib/libpython3.9.so
../lib/orbslam2.so: /home/main/lihy/RTG-SLAM/thirdParty/install/lib/libopencv_imgcodecs.so.4.2.0
../lib/orbslam2.so: /home/main/lihy/RTG-SLAM/thirdParty/install/lib/libopencv_calib3d.so.4.2.0
../lib/orbslam2.so: /home/main/lihy/RTG-SLAM/thirdParty/install/lib/libopencv_features2d.so.4.2.0
../lib/orbslam2.so: /home/main/lihy/RTG-SLAM/thirdParty/install/lib/libopencv_flann.so.4.2.0
../lib/orbslam2.so: /home/main/lihy/RTG-SLAM/thirdParty/install/lib/libopencv_imgproc.so.4.2.0
../lib/orbslam2.so: /home/main/lihy/RTG-SLAM/thirdParty/install/lib/libopencv_core.so.4.2.0
../lib/orbslam2.so: CMakeFiles/orbslam2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/main/lihy/RTG-SLAM/thirdParty/pybind/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library ../lib/orbslam2.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/orbslam2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/orbslam2.dir/build: ../lib/orbslam2.so
.PHONY : CMakeFiles/orbslam2.dir/build

CMakeFiles/orbslam2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/orbslam2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/orbslam2.dir/clean

CMakeFiles/orbslam2.dir/depend:
	cd /home/main/lihy/RTG-SLAM/thirdParty/pybind/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/main/lihy/RTG-SLAM/thirdParty/pybind /home/main/lihy/RTG-SLAM/thirdParty/pybind /home/main/lihy/RTG-SLAM/thirdParty/pybind/build /home/main/lihy/RTG-SLAM/thirdParty/pybind/build /home/main/lihy/RTG-SLAM/thirdParty/pybind/build/CMakeFiles/orbslam2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/orbslam2.dir/depend

