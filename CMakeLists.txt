cmake_minimum_required(VERSION 3.21)
project(HIP_GPT LANGUAGES CXX)

# HIP setup
find_package(HIP REQUIRED)

# Compiler flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
file(GLOB SRC
    "*.cpp"
    "hip_kernels.cpp"
)

# Executable
add_executable(train_gpt ${SRC})

# Include directories
target_include_directories(train_gpt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link HIP
target_link_libraries(train_gpt PRIVATE HIP::hiprtc HIP::device HIP::host)

# Optional: Enable verbose output
set(CMAKE_VERBOSE_MAKEFILE ON)

add_executable(generate generate.cpp)
target_link_libraries(generate PRIVATE HIP::hiprtc HIP::device HIP::host)

