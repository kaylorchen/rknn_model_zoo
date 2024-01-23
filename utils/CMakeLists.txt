cmake_minimum_required(VERSION 3.15)

project(rknn_model_zoo_utils)

add_library(fileutils STATIC
    file_utils.c
)
target_include_directories(fileutils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(imagedrawing STATIC
    image_drawing.c
)
target_include_directories(imagedrawing PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# only RGA on rv1106 and rk3588 support handle
if (TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rk3588")
    add_definitions(-DLIBRGA_IM2D_HANDLE)
endif()

set(ENV{PKG_CONFIG_SYSROOT_DIR} ${CMAKE_SYSROOT})
set(ENV{PKG_CONFIG_PATH} ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig)

find_package(PkgConfig REQUIRED)
# Use pkg-config to look for librga
pkg_check_modules(LIBRGA REQUIRED librga)
pkg_check_modules(STB REQUIRED stb)
pkg_check_modules(TURBOJPEG REQUIRED libturbojpeg)

include_directories(${LIBRGA_INCLUDE_DIRS})
link_directories(${LIBRGA_LIBRARY_DIRS})
link_directories(${STB_LIBRARY_DIRS})

add_library(imageutils STATIC
    image_utils.c
)
target_include_directories(imageutils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(imageutils
    ${TURBOJPEG_LIBRARIES}
    ${LIBRGA_LIBRARIES}
    ${STB_LIBRARIES}
)
target_include_directories(imageutils PUBLIC
    ${STB_INCLUDES}
    ${TURBOJPEG_INCLUDES}
    ${LIBRGA_INCLUDES}
)
