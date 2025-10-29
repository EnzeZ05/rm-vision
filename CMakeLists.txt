cmake_minimum_required(VERSION 3.30)
project(rm_vision)

set(CMAKE_CXX_STANDARD 20)

# OpenCV
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/lib/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Torch
set(Torch_DIR "${CMAKE_SOURCE_DIR}/lib/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

include_directories("include")

add_executable(${PROJECT_NAME}
        src/main.cpp
        src/Armor.cpp
)
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

# Copy runtime environments to package
add_custom_command(TARGET rm_vision POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${OpenCV_DIR}/x64/vc16/bin" $<TARGET_FILE_DIR:rm_vision>
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/lib/libtorch/lib" $<TARGET_FILE_DIR:rm_vision>
)
