# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

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
CMAKE_SOURCE_DIR = /home/lxr/sc2023/SC2023/cpu

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lxr/sc2023/SC2023/cpu/build

# Include any dependencies generated for this target.
include CMakeFiles/dawn_cpu_sssp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/dawn_cpu_sssp.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/dawn_cpu_sssp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dawn_cpu_sssp.dir/flags.make

CMakeFiles/dawn_cpu_sssp.dir/dawn_cpu_sssp.cpp.o: CMakeFiles/dawn_cpu_sssp.dir/flags.make
CMakeFiles/dawn_cpu_sssp.dir/dawn_cpu_sssp.cpp.o: /home/lxr/sc2023/SC2023/cpu/dawn_cpu_sssp.cpp
CMakeFiles/dawn_cpu_sssp.dir/dawn_cpu_sssp.cpp.o: CMakeFiles/dawn_cpu_sssp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lxr/sc2023/SC2023/cpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/dawn_cpu_sssp.dir/dawn_cpu_sssp.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/dawn_cpu_sssp.dir/dawn_cpu_sssp.cpp.o -MF CMakeFiles/dawn_cpu_sssp.dir/dawn_cpu_sssp.cpp.o.d -o CMakeFiles/dawn_cpu_sssp.dir/dawn_cpu_sssp.cpp.o -c /home/lxr/sc2023/SC2023/cpu/dawn_cpu_sssp.cpp

CMakeFiles/dawn_cpu_sssp.dir/dawn_cpu_sssp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dawn_cpu_sssp.dir/dawn_cpu_sssp.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lxr/sc2023/SC2023/cpu/dawn_cpu_sssp.cpp > CMakeFiles/dawn_cpu_sssp.dir/dawn_cpu_sssp.cpp.i

CMakeFiles/dawn_cpu_sssp.dir/dawn_cpu_sssp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dawn_cpu_sssp.dir/dawn_cpu_sssp.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lxr/sc2023/SC2023/cpu/dawn_cpu_sssp.cpp -o CMakeFiles/dawn_cpu_sssp.dir/dawn_cpu_sssp.cpp.s

# Object files for target dawn_cpu_sssp
dawn_cpu_sssp_OBJECTS = \
"CMakeFiles/dawn_cpu_sssp.dir/dawn_cpu_sssp.cpp.o"

# External object files for target dawn_cpu_sssp
dawn_cpu_sssp_EXTERNAL_OBJECTS =

dawn_cpu_sssp: CMakeFiles/dawn_cpu_sssp.dir/dawn_cpu_sssp.cpp.o
dawn_cpu_sssp: CMakeFiles/dawn_cpu_sssp.dir/build.make
dawn_cpu_sssp: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
dawn_cpu_sssp: /usr/lib/x86_64-linux-gnu/libpthread.so
dawn_cpu_sssp: CMakeFiles/dawn_cpu_sssp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lxr/sc2023/SC2023/cpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable dawn_cpu_sssp"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dawn_cpu_sssp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dawn_cpu_sssp.dir/build: dawn_cpu_sssp
.PHONY : CMakeFiles/dawn_cpu_sssp.dir/build

CMakeFiles/dawn_cpu_sssp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dawn_cpu_sssp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dawn_cpu_sssp.dir/clean

CMakeFiles/dawn_cpu_sssp.dir/depend:
	cd /home/lxr/sc2023/SC2023/cpu/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lxr/sc2023/SC2023/cpu /home/lxr/sc2023/SC2023/cpu /home/lxr/sc2023/SC2023/cpu/build /home/lxr/sc2023/SC2023/cpu/build /home/lxr/sc2023/SC2023/cpu/build/CMakeFiles/dawn_cpu_sssp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dawn_cpu_sssp.dir/depend

