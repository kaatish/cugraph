#=============================================================================
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)
include(FetchContent)
FetchContent_Declare(
  rapids-cmake
  GIT_REPOSITORY https://github.com/rapidsai/rapids-cmake.git
  GIT_TAG        origin/branch-21.06
  )
FetchContent_MakeAvailable(rapids-cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(CUGRAPH)

project(CUGRAPH VERSION 21.08.00 LANGUAGES C CXX CUDA)

# Remove the following archs from CMAKE_CUDA_ARCHITECTURES that
# cuhornet currently doesn't support
#
# < 60
# >= 86
set(supported_archs "60" "62" "70" "72" "75" "80")
foreach( arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REPLACE "-real" "" arch ${arch})
    if( arch IN_LIST supported_archs )
        list(APPEND usable_arch_values ${arch})
    endif()
endforeach()
# Make sure everything but the 'newest' arch
# is marked as `-real` so we only generate PTX for
# arch > 80
list(POP_BACK usable_arch_values latest_arch)
list(TRANSFORM usable_arch_values APPEND "-real")
list(APPEND usable_arch_values ${latest_arch})

set(CMAKE_CUDA_ARCHITECTURES ${usable_arch_values})


# Write the version header
include(cmake/Modules/Version.cmake)
write_version()

###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

##############################################################################
# - User Options  ------------------------------------------------------------

option(BUILD_CUGRAPH_MG_TESTS "Build cuGraph multigpu algorithm tests" OFF)
set(BLAS_LIBRARIES "" CACHE STRING "Location of BLAS library for FAISS build.")
option(BUILD_STATIC_FAISS "Build the FAISS library for nearest neighbors search on GPU" OFF)
option(CMAKE_CUDA_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
option(BUILD_TESTS "Configure CMake to build tests" ON)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

rapids_find_package(CUDAToolkit REQUIRED
    BUILD_EXPORT_SET    cugraph-exports
    INSTALL_EXPORT_SET  cugraph-exports
    )

set(CUGRAPH_CXX_FLAGS "")
set(CUGRAPH_CUDA_FLAGS "")

if(CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND CUGRAPH_CXX_FLAGS -Werror -Wno-error=deprecated-declarations)
endif(CMAKE_COMPILER_IS_GNUCXX)


message("-- Building for GPU_ARCHS = ${CMAKE_CUDA_ARCHITECTURES}")

list(APPEND CUGRAPH_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
list(APPEND CUGRAPH_CUDA_FLAGS -Werror=cross-execution-space-call -Wno-deprecated-declarations -Xptxas=--disable-warnings)
list(APPEND CUGRAPH_CUDA_FLAGS -Xcompiler=-Wall,-Wno-error=sign-compare,-Wno-error=unused-but-set-variable)
list(APPEND CUGRAPH_CUDA_FLAGS -Xfatbin=-compress-all)

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
if (CMAKE_CUDA_LINEINFO)
    list(APPEND CUGRAPH_CUDA_FLAGS -lineinfo)
endif(CMAKE_CUDA_LINEINFO)

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    list(APPEND CUGRAPH_CUDA_FLAGS -G -Xcompiler=-rdynamic)
endif(CMAKE_BUILD_TYPE MATCHES Debug)

###################################################################################################
# - find openmp -----------------------------------------------------------------------------------

find_package(OpenMP)
if(OpenMP_FOUND)
    # find_package(OPenMP) does not automatically add OpenMP flags to CUDA
    list(APPEND CUGRAPH_CUDA_FLAGS -Xcompiler=${OpenMP_CXX_FLAGS})
endif(OpenMP_FOUND)


###################################################################################################
# - find blas -------------------------------------------------------------------------------------

if(NOT DEFINED BLAS_LIBRARIES)
  find_package( BLAS REQUIRED )
else()
  message(STATUS "Manually setting BLAS to ${BLAS_LIBRARIES}")
endif()

###################################################################################################
# - find CPM based dependencies  ------------------------------------------------------------------


rapids_cpm_init()


include(cmake/thirdparty/get_thrust.cmake)
include(cmake/thirdparty/get_faiss.cmake)
include(cmake/thirdparty/get_nccl.cmake)
include(cmake/thirdparty/get_rmm.cmake)

include(cmake/thirdparty/get_raft.cmake)

include(cmake/thirdparty/get_cuco.cmake)
include(cmake/thirdparty/get_cuhornet.cmake)

include(cmake/thirdparty/get_gunrock.cmake)

if(BUILD_TESTS)
  include(cmake/thirdparty/get_gtest.cmake)
endif()


###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(cugraph SHARED
    src/utilities/spmv_1D.cu
    src/utilities/cython.cu
    src/utilities/path_retrieval.cu
    src/utilities/graph_bcast.cu
    src/structure/graph.cu
    src/linear_assignment/hungarian.cu
    src/link_analysis/gunrock_hits.cpp
    src/traversal/bfs.cu
    src/traversal/sssp.cu
    src/traversal/tsp.cu
    src/link_prediction/jaccard.cu
    src/link_prediction/overlap.cu
    src/layout/force_atlas2.cu
    src/converters/renumber.cu
    src/converters/COOtoCSR.cu
    src/community/spectral_clustering.cu
    src/community/louvain.cu
    src/community/leiden.cu
    src/community/ktruss.cu
    src/community/ecg.cu
    src/community/triangles_counting.cu
    src/community/extract_subgraph_by_vertex.cu
    src/community/egonet.cu
    src/sampling/random_walks.cu
    src/cores/core_number.cu
    src/traversal/two_hop_neighbors.cu
    src/components/connectivity.cu
    src/centrality/katz_centrality.cu
    src/centrality/betweenness_centrality.cu
    src/generators/generate_rmat_edgelist.cu
    src/generators/generator_tools.cu
    src/generators/simple_generators.cu
    src/generators/erdos_renyi_generator.cu
    src/experimental/graph.cu
    src/experimental/graph_view.cu
    src/experimental/coarsen_graph.cu
    src/experimental/renumber_edgelist.cu
    src/experimental/renumber_utils.cu
    src/experimental/relabel.cu
    src/experimental/induced_subgraph.cu
    src/experimental/bfs.cu
    src/experimental/sssp.cu
    src/experimental/pagerank.cu
    src/experimental/katz_centrality.cu
    src/serialization/serializer.cu
    src/tree/mst.cu
    src/components/weakly_connected_components.cu
    src/structure/create_graph_from_edgelist.cu
    src/utilities/host_barrier.cpp
)

set_target_properties(cugraph
    PROPERTIES BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               # set target compile options
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       17
               CUDA_STANDARD_REQUIRED              ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(cugraph
            PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_CXX_FLAGS}>"
                    "$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_CUDA_FLAGS}>"
)

# Per-thread default stream option see https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html
# The per-thread default stream does not synchronize with other streams
target_compile_definitions(cugraph PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)

file(WRITE "${CUGRAPH_BINARY_DIR}/fatbin.ld"
[=[
SECTIONS
{
  .nvFatBinSegment : { *(.nvFatBinSegment) }
  .nv_fatbin : { *(.nv_fatbin) }
}
]=])
target_link_options(cugraph PRIVATE "${CUGRAPH_BINARY_DIR}/fatbin.ld")

add_library(cugraph::cugraph ALIAS cugraph)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------
target_include_directories(cugraph
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${NCCL_INCLUDE_DIRS}"
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

###################################################################################################
# - link libraries --------------------------------------------------------------------------------
target_link_libraries(cugraph
    PUBLIC
        rmm::rmm
        cugraph::Thrust
        raft::raft
    PRIVATE
        cuco::cuco
        CUDA::cublas
        CUDA::curand
        CUDA::cusolver
        CUDA::cusparse
        cugraph::cuHornet
        FAISS::FAISS
        gunrock
        NCCL::NCCL
)

if(OpenMP_CXX_FOUND)
target_link_libraries(cugraph PRIVATE
###################################################################################################
###   Use ${OpenMP_CXX_LIB_NAMES} instead of OpenMP::OpenMP_CXX to avoid the following warnings.
###
###   Cannot generate a safe runtime search path for target TARGET_NAME
###   because files in some directories may conflict with libraries in implicit
###   directories:
###   ...
###
###   libgomp.so is included in the conda base environment and copied to every new conda
###   environment. If a full file path is provided (e.g ${NCCL_LIBRARIES}), cmake
###   extracts the directory path and adds the directory path to BUILD_RPATH (if BUILD_RPATH is not
###   disabled).
###
###   cmake maintains a system specific implicit directories (e.g. /lib, /lib/x86_64-linux-gnu,
###   /lib32, /lib32/x86_64-linux-gnu, /lib64, /lib64/x86_64-linux-gnu, /usr/lib,
###   /usr/lib/gcc/x86_64-linux-gnu/7, /usr/lib/x86_64-linux-gnu, /usr/lib32,
###   /usr/lib32/x86_64-linux-gnu, /usr/lib64, /usr/lib64/x86_64-linux-gnu,
###   /usr/local/cuda-10.0/lib64", /usr/local/cuda-10.0/lib64/stubs).
###
###   If a full path to libgomp.so is provided (which is the case with OpenMP::OpenMP_CXX), cmake
###   checks whether there is any other libgomp.so with the different full path (after resolving
###   soft links) in the search paths (implicit directoires + BUILD_RAPTH). There is one in the
###   path included in BUILD_RPATH when ${NCCL_LIBRARIES} are added; this one can
###   potentially hide the one in the provided full path and cmake generates a warning (and RPATH
###   is searched before the directories in /etc/ld.so/conf; ld.so.conf does not coincide but
###   overlaps with implicit directories).
###
###   If we provide just the library names (gomp;pthread), cmake does not generate warnings (we
###   did not specify which libgomp.so should be loaded in runtime), and the one first found in
###   the search order is loaded (we can change the loaded library by setting LD_LIBRARY_PATH or
###   manually editing BUILD_RPATH).
###
###   Manually editing BUILD_RPATH:
###   set(TARGET_BUILD_RPATH "")
###   foreach(TMP_VAR_FULLPATH IN LISTS OpenMP_CXX_LIBRARIES)
###       get_filename_component(TMP_VAR_DIR ${TMP_VAR_FULLPATH} DIRECTORY)
###       string(APPEND TARGET_BUILD_RPATH "${TMP_VAR_DIR};")
###       get_filename_component(TMP_VAR_REALPATH ${TMP_VAR_FULLPATH} REALPATH)
###       get_filename_component(TMP_VAR_DIR ${TMP_VAR_REALPATH} DIRECTORY)
###       # cmake automatically removes duplicates, so skip checking.
###       string(APPEND TARGET_BUILD_RPATH "${TMP_VAR_DIR};")
###   endforeach()
###   string(APPEND TARGET_BUILD_RPATH "${CONDA_PREFIX}/lib")
###   message(STATUS "TARGET_BUILD_RPATH=${TARGET_BUILD_RPATH}")
###   set_target_properties(target PROPERTIES
###       BUILD_RPATH "${TARGET_BUILD_RPATH}")
    ${OpenMP_CXX_LIB_NAMES})
endif(OpenMP_CXX_FOUND)


###################################################################################################
# - generate tests --------------------------------------------------------------------------------

if(BUILD_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif(BUILD_TESTS)

###################################################################################################
# - install targets -------------------------------------------------------------------------------

install(TARGETS cugraph
        DESTINATION lib
        EXPORT cugraph-exports)

install(DIRECTORY include/cugraph/
        DESTINATION include/cugraph)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/cugraph/version_config.hpp
        DESTINATION include/cugraph)

################################################################################################
# - install export -------------------------------------------------------------------------------
set(doc_string
[=[
Provide targets for cuGraph.

cuGraph library is a collection of GPU accelerated graph algorithms that process data found in
[GPU DataFrames](https://github.com/rapidsai/cudf).

]=])

set(code_string
[=[
thrust_create_target(cugraph::Thrust FROM_OPTIONS)
]=])

 rapids_export(INSTALL cugraph
    EXPORT_SET cugraph-exports
    GLOBAL_TARGETS cugraph
    NAMESPACE cugraph::
    DOCUMENTATION doc_string
    FINAL_CODE_BLOCK code_string
    )

################################################################################################
# - build export -------------------------------------------------------------------------------
rapids_export(BUILD cugraph
    EXPORT_SET cugraph-exports
    GLOBAL_TARGETS cugraph
    NAMESPACE cugraph::
    DOCUMENTATION doc_string
    FINAL_CODE_BLOCK code_string
    )

###################################################################################################
# - make documentation ----------------------------------------------------------------------------
# requires doxygen and graphviz to be installed
# from build directory, run make docs_cugraph

# doc targets for cugraph
find_package(Doxygen 1.8.11)
if(Doxygen_FOUND)
    add_custom_command(OUTPUT CUGRAPH_DOXYGEN
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
                       COMMAND doxygen Doxyfile
                       VERBATIM)

    add_custom_target(docs_cugraph DEPENDS CUGRAPH_DOXYGEN)
endif()
