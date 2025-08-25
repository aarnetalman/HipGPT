cmake_minimum_required(VERSION 3.21)
# Renamed project as requested
project(HipGPT LANGUAGES CXX HIP)

# --- Dependency Management ---
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(nlohmann_json)

# --- HIP & Compiler Setup ---
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

# --- Source Files ---
file(GLOB GPT_SOURCES "src/*.cpp")
list(FILTER GPT_SOURCES EXCLUDE REGEX ".*/(train_gpt|generate)\\.cpp$")
set(TRAIN_MAIN_SRC src/train_gpt.cpp)
set(GENERATE_MAIN_SRC src/generate.cpp)

# --- Executables ---
add_executable(train_gpt ${TRAIN_MAIN_SRC} ${GPT_SOURCES})
add_executable(generate ${GENERATE_MAIN_SRC} ${GPT_SOURCES})

# --- Include Directories ---
target_include_directories(train_gpt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${nlohmann_json_SOURCE_DIR}/include
    ${HIP_INCLUDE_DIRS} 
)
target_include_directories(generate PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${nlohmann_json_SOURCE_DIR}/include
    ${HIP_INCLUDE_DIRS} 
)

# Optional: Enable verbose output
set(CMAKE_VERBOSE_MAKEFILE ON)