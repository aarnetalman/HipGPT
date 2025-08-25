cmake_minimum_required(VERSION 3.21)
# Add HIP as a project language
project(HIP_GPT LANGUAGES CXX HIP)

# --- Dependency Management ---
# Automatically fetch the nlohmann-json library
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(nlohmann_json)


# --- HIP & Compiler Setup ---
find_package(HIP REQUIRED)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

# --- Source Files ---
set(GPT_SOURCES
    gpt_model.cpp
    transformer_layer.cpp
    hip_kernels.cpp
    tokenizer.cpp
)

# --- Executables ---
add_executable(train_gpt
    train_gpt.cpp
    ${GPT_SOURCES}
)

add_executable(generate
    generate.cpp
    ${GPT_SOURCES}
)

# --- Include Directories ---
target_include_directories(train_gpt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    # Add the include directory for the downloaded json library
    ${nlohmann_json_SOURCE_DIR}/include
)
target_include_directories(generate PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    # Add the include directory for the downloaded json library
    ${nlohmann_json_SOURCE_DIR}/include
)

# --- Link Libraries ---
# Note: When HIP is a project language, linking is often handled automatically.
# Explicitly linking is still safe.
target_link_libraries(train_gpt PRIVATE HIP::hiprtc HIP::device HIP::host)
target_link_libraries(generate PRIVATE HIP::hiprtc HIP::device HIP::host)

# Optional: Enable verbose output
set(CMAKE_VERBOSE_MAKEFILE ON)