cmake_minimum_required(VERSION 3.21)
project(HIP_GPT LANGUAGES CXX HIP)

# --- Dependency Management ---
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
# Use a glob to find all .cpp files in the src directory
file(GLOB GPT_SOURCES "src/*.cpp")

# Separate the main files for the executables
list(FILTER GPT_SOURCES EXCLUDE REGEX ".*/(train_gpt|generate)\\.cpp$")
set(TRAIN_MAIN_SRC src/train_gpt.cpp)
set(GENERATE_MAIN_SRC src/generate.cpp)

# --- Executables ---
add_executable(train_gpt ${TRAIN_MAIN_SRC} ${GPT_SOURCES})
add_executable(generate ${GENERATE_MAIN_SRC} ${GPT_SOURCES})

# --- Include Directories ---
# Add the 'include' and the downloaded json library directories
target_include_directories(train_gpt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${nlohmann_json_SOURCE_DIR}/include
)
target_include_directories(generate PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${nlohmann_json_SOURCE_DIR}/include
)

# --- Link Libraries ---
target_link_libraries(train_gpt PRIVATE HIP::hiprtc HIP::device HIP::host)
target_link_libraries(generate PRIVATE HIP::hiprtc HIP::device HIP::host)