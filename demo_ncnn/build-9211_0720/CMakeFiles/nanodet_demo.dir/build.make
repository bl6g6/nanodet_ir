# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

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
CMAKE_COMMAND = /opt/cmake-3.20.2-linux-x86_64/bin/cmake

# The command to remove a file.
RM = /opt/cmake-3.20.2-linux-x86_64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/masike/server_masike/nanodet/demo_ncnn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/masike/server_masike/nanodet/demo_ncnn/build-9211_0720

# Include any dependencies generated for this target.
include CMakeFiles/nanodet_demo.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/nanodet_demo.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/nanodet_demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/nanodet_demo.dir/flags.make

CMakeFiles/nanodet_demo.dir/main.cpp.o: CMakeFiles/nanodet_demo.dir/flags.make
CMakeFiles/nanodet_demo.dir/main.cpp.o: ../main.cpp
CMakeFiles/nanodet_demo.dir/main.cpp.o: CMakeFiles/nanodet_demo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/masike/server_masike/nanodet/demo_ncnn/build-9211_0720/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/nanodet_demo.dir/main.cpp.o"
	/home/masike/mylib/toolChain/9211_linux/gcc-sigmastar-9.1.0-2020.07-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/nanodet_demo.dir/main.cpp.o -MF CMakeFiles/nanodet_demo.dir/main.cpp.o.d -o CMakeFiles/nanodet_demo.dir/main.cpp.o -c /home/masike/server_masike/nanodet/demo_ncnn/main.cpp

CMakeFiles/nanodet_demo.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nanodet_demo.dir/main.cpp.i"
	/home/masike/mylib/toolChain/9211_linux/gcc-sigmastar-9.1.0-2020.07-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/masike/server_masike/nanodet/demo_ncnn/main.cpp > CMakeFiles/nanodet_demo.dir/main.cpp.i

CMakeFiles/nanodet_demo.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nanodet_demo.dir/main.cpp.s"
	/home/masike/mylib/toolChain/9211_linux/gcc-sigmastar-9.1.0-2020.07-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/masike/server_masike/nanodet/demo_ncnn/main.cpp -o CMakeFiles/nanodet_demo.dir/main.cpp.s

CMakeFiles/nanodet_demo.dir/nanodet.cpp.o: CMakeFiles/nanodet_demo.dir/flags.make
CMakeFiles/nanodet_demo.dir/nanodet.cpp.o: ../nanodet.cpp
CMakeFiles/nanodet_demo.dir/nanodet.cpp.o: CMakeFiles/nanodet_demo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/masike/server_masike/nanodet/demo_ncnn/build-9211_0720/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/nanodet_demo.dir/nanodet.cpp.o"
	/home/masike/mylib/toolChain/9211_linux/gcc-sigmastar-9.1.0-2020.07-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/nanodet_demo.dir/nanodet.cpp.o -MF CMakeFiles/nanodet_demo.dir/nanodet.cpp.o.d -o CMakeFiles/nanodet_demo.dir/nanodet.cpp.o -c /home/masike/server_masike/nanodet/demo_ncnn/nanodet.cpp

CMakeFiles/nanodet_demo.dir/nanodet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nanodet_demo.dir/nanodet.cpp.i"
	/home/masike/mylib/toolChain/9211_linux/gcc-sigmastar-9.1.0-2020.07-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/masike/server_masike/nanodet/demo_ncnn/nanodet.cpp > CMakeFiles/nanodet_demo.dir/nanodet.cpp.i

CMakeFiles/nanodet_demo.dir/nanodet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nanodet_demo.dir/nanodet.cpp.s"
	/home/masike/mylib/toolChain/9211_linux/gcc-sigmastar-9.1.0-2020.07-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/masike/server_masike/nanodet/demo_ncnn/nanodet.cpp -o CMakeFiles/nanodet_demo.dir/nanodet.cpp.s

# Object files for target nanodet_demo
nanodet_demo_OBJECTS = \
"CMakeFiles/nanodet_demo.dir/main.cpp.o" \
"CMakeFiles/nanodet_demo.dir/nanodet.cpp.o"

# External object files for target nanodet_demo
nanodet_demo_EXTERNAL_OBJECTS =

nanodet_demo: CMakeFiles/nanodet_demo.dir/main.cpp.o
nanodet_demo: CMakeFiles/nanodet_demo.dir/nanodet.cpp.o
nanodet_demo: CMakeFiles/nanodet_demo.dir/build.make
nanodet_demo: /home/masike/prefix/9211_linux/ncnn/0720_nostd/lib/libncnn.a
nanodet_demo: /usr/local/lib/libopencv_highgui.so.4.5.2
nanodet_demo: /usr/local/lib/libopencv_ml.so.4.5.2
nanodet_demo: /usr/local/lib/libopencv_objdetect.so.4.5.2
nanodet_demo: /usr/local/lib/libopencv_photo.so.4.5.2
nanodet_demo: /usr/local/lib/libopencv_stitching.so.4.5.2
nanodet_demo: /usr/local/lib/libopencv_video.so.4.5.2
nanodet_demo: /usr/local/lib/libopencv_videoio.so.4.5.2
nanodet_demo: /home/masike/mylib/toolChain/9211_linux/gcc-sigmastar-9.1.0-2020.07-x86_64_arm-linux-gnueabihf/arm-linux-gnueabihf/lib/libgomp.so
nanodet_demo: /home/masike/mylib/toolChain/9211_linux/gcc-sigmastar-9.1.0-2020.07-x86_64_arm-linux-gnueabihf/arm-linux-gnueabihf/libc/usr/lib/libpthread.so
nanodet_demo: /usr/local/lib/libopencv_imgcodecs.so.4.5.2
nanodet_demo: /usr/local/lib/libopencv_calib3d.so.4.5.2
nanodet_demo: /usr/local/lib/libopencv_dnn.so.4.5.2
nanodet_demo: /usr/local/lib/libopencv_features2d.so.4.5.2
nanodet_demo: /usr/local/lib/libopencv_flann.so.4.5.2
nanodet_demo: /usr/local/lib/libopencv_imgproc.so.4.5.2
nanodet_demo: /usr/local/lib/libopencv_core.so.4.5.2
nanodet_demo: CMakeFiles/nanodet_demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/masike/server_masike/nanodet/demo_ncnn/build-9211_0720/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable nanodet_demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nanodet_demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/nanodet_demo.dir/build: nanodet_demo
.PHONY : CMakeFiles/nanodet_demo.dir/build

CMakeFiles/nanodet_demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/nanodet_demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/nanodet_demo.dir/clean

CMakeFiles/nanodet_demo.dir/depend:
	cd /home/masike/server_masike/nanodet/demo_ncnn/build-9211_0720 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/masike/server_masike/nanodet/demo_ncnn /home/masike/server_masike/nanodet/demo_ncnn /home/masike/server_masike/nanodet/demo_ncnn/build-9211_0720 /home/masike/server_masike/nanodet/demo_ncnn/build-9211_0720 /home/masike/server_masike/nanodet/demo_ncnn/build-9211_0720/CMakeFiles/nanodet_demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/nanodet_demo.dir/depend

