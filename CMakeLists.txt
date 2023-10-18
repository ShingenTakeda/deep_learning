cmake_minimum_required(VERSION 3.15)
project(deep_learning)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 CONFIG REQUIRED)
find_package(sciplot CONFIG REQUIRED)

include_directories(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/)
file(GLOB_RECURSE N_SRC "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*.hpp")

add_executable(${PROJECT_NAME} ${N_SRC})
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen sciplot::sciplot)
