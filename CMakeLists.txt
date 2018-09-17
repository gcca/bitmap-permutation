cmake_minimum_required(VERSION 3.11)

find_package(Threads)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 11)

include(ExternalProject)

if(NOT GTEST_ROOT)
ExternalProject_Add(googletest_ep
	CMAKE_ARGS
		-DCMAKE_BUILD_TYPE=RELEASE
		-DCMAKE_INSTALL_PREFIX=build
	GIT_REPOSITORY git@github.com:google/googletest.git
	UPDATE_COMMAND "")
ExternalProject_Get_property(googletest_ep BINARY_DIR)
set(GTEST_ROOT ${BINARY_DIR}/build)

file(MAKE_DIRECTORY ${GTEST_ROOT}/include)
file(MAKE_DIRECTORY ${GTEST_ROOT}/lib)

endif()

add_library(GTest::GTest INTERFACE IMPORTED)
add_dependencies(GTest::GTest googletest_ep)
target_include_directories(GTest::GTest INTERFACE ${GTEST_ROOT}/include)
target_link_libraries(GTest::GTest
    INTERFACE ${GTEST_ROOT}/lib/libgtest.a Threads::Threads)

add_library(GTest::Main INTERFACE IMPORTED)
add_dependencies(GTest::Main GTest::GTest)
target_link_libraries(GTest::Main INTERFACE ${GTEST_ROOT}/lib/libgtest_main.a)

enable_testing()

add_executable(bitmap_permutation-test bitmap_permutation-test.cu)
target_link_libraries(bitmap_permutation-test GTest::GTest GTest::Main)
