cmake_minimum_required(VERSION 3.1)
project(ACAP)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)

find_package(LIBIGL REQUIRED QUIET)

# Cplex
find_package(CPLEX REQUIRED)
if (CPLEX_FOUND)
    include_directories(${CPLEX_INCLUDE_DIR})
    link_directories(${CPLEX_LIBRARY})
endif ()

if (CPLEX_CONCERT_FOUND)
    include_directories(${CPLEX_CONCERT_INCLUDE_DIR})
    link_directories(${CPLEX_CONCERT_LIBRARY})
endif ()

if (CPLEX_ILOCPLEX_FOUND)
    include_directories(${CPLEX_ILOCPLEX_INCLUDE_DIR})
    link_directories(${CPLEX_ILOCPLEX_LIBRARY})
endif ()


# Add your project files
file(GLOB SRCFILES ./src/*.cpp)
add_executable(${PROJECT_NAME}_bin ${SRCFILES})
target_link_libraries(${PROJECT_NAME}_bin igl::core igl::opengl_glfw cplex-library cplex-concert ilocplex)
#target_link_libraries(${PROJECT_NAME}_bin igl::core igl::opengl_glfw)