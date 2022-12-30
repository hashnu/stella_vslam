# ----- Find dependencies -----

# popl
set(popl_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/3rd/popl/include)

set(USE_STACK_TRACE_LOGGER OFF CACHE BOOL "Enable automatic stack trace logger of google-glog")

# ----- Build example executables -----
set(EXECUTABLE_TARGETS "")

find_package(JlCxx)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location}")
message(STATUS "Found JlCxx at ${JlCxx_location}")

# add_executable(run_camera_slam run_camera_slam.cc)
# list(APPEND EXECUTABLE_TARGETS run_camera_slam)

# add_executable(run_camera_localization run_camera_localization.cc)
# list(APPEND EXECUTABLE_TARGETS run_camera_localization)
add_library(stella_vslam_jll SHARED stella_slam_jll.cc)
list(APPEND EXECUTABLE_TARGETS stella_vslam_jll)

foreach(EXECUTABLE_TARGET IN LISTS EXECUTABLE_TARGETS)
    # Set output directory for executables
    set_target_properties(${EXECUTABLE_TARGET} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${PROJECT_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${PROJECT_BINARY_DIR}")

    # Link PLPSLAM
    target_link_libraries(${EXECUTABLE_TARGET}
        PRIVATE
        ${PROJECT_NAME}
        opencv_imgcodecs
        opencv_videoio
        JlCxx::cxxwrap_julia)

    # include popl and spdlog headers
    target_include_directories(${EXECUTABLE_TARGET}
        PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/3rd/popl/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/3rd/spdlog/include>)
endforeach()

#cmake -DBUILD_JULIA=ON -DCMAKE_BUILD_TYPE=Release -DJlCxx_DIR=/opt/neleus_src/libcxxwrap-julia/build/1.8.1 -DJulia_EXECUTABLE=/opt/julia-1.8.1/bin/julia -DJULIA_PKG_INSTALL_PATH=/opt/juliaDepot/packages ..
#cmake -DUSE_OPENMP=ON -DUSE_SSE_FP_MATH=ON -DUSE_SSE_ORB=ON -DBUILD_WITH_MARCH_NATIVE=OFF -D OpenGL_GL_PREFERENCE=GLVND ..
