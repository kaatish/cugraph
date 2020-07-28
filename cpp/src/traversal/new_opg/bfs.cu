/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

#include <algorithm>
#include <iomanip>
#include <limits>
#include "bfs.cuh"

#include "graph.hpp"

#include <utilities/error.hpp>
#include "bfs_kernels.cuh"
#include "opg/bfs.cuh"
#include "traversal_common.cuh"
#include "utilities/graph_utils.cuh"

namespace cugraph {
namespace detail {
enum BFS_ALGO_STATE { TOPDOWN, BOTTOMUP };

template <typename IndexType>
void BFS<IndexType>::setup()
{

  // Working data
  // Each vertex can be in the frontier at most once
  // We will update frontier during the execution
  // We need the orig to reset frontier, or ALLOC_FREE_TRY
  original_frontier.resize(number_of_vertices);
  frontier = original_frontier.data().get();

  // size of bitmaps for vertices
  vertices_bmap_size = (number_of_vertices / (8 * sizeof(int)) + 1);
  // ith bit of visited_bmap is set <=> ith vertex is visited

  visited_bmap.resize(vertices_bmap_size);

  // ith bit of isolated_bmap is set <=> degree of ith vertex = 0
  isolated_bmap.resize(vertices_bmap_size);

  // vertices_degree[i] = degree of vertex i
  vertex_degree.resize(number_of_vertices);

  // We will need (n+1) ints buffer for two differents things (bottom up or top down) - sharing it
  // since those uses are mutually exclusive
  buffer_np1_1.resize(number_of_vertices + 1);
  buffer_np1_2.resize(number_of_vertices + 1);

  // Using buffers : top down

  // frontier_vertex_degree[i] is the degree of vertex frontier[i]
  frontier_vertex_degree = buffer_np1_1.data().get();
  // exclusive sum of frontier_vertex_degree
  exclusive_sum_frontier_vertex_degree = buffer_np1_2.data().get();

  // Using buffers : bottom up
  // contains list of unvisited vertices
  unvisited_queue = buffer_np1_1.data().get();
  // size of the "last" unvisited queue : size_last_unvisited_queue
  // refers to the size of unvisited_queue
  // which may not be up to date (the queue may contains vertices that are now
  // visited)

  // We may leave vertices unvisited after bottom up main kernels - storing them
  // here
  left_unvisited_queue = buffer_np1_2.data().get();

  // We use buckets of edges (32 edges per bucket for now, see exact macro in bfs_kernels).
  // frontier_vertex_degree_buckets_offsets[i] is the index k such as frontier[k] is the source of
  // the first edge of the bucket See top down kernels for more details
  exclusive_sum_frontier_vertex_buckets_offsets.resize(
    ((number_of_edges / TOP_DOWN_EXPAND_DIMX + 1) * NBUCKETS_PER_BLOCK + 2));

  // Init device-side counters
  // Those counters must be/can be reset at each bfs iteration
  // Keeping them adjacent in memory allow use call only one cudaMemset - launch latency is the
  // current bottleneck
  d_counters_pad.resize(4);

  d_new_frontier_cnt   = d_counters_pad.data().get();
  d_mu                 = d_counters_pad.data().get() + 1;
  d_unvisited_cnt      = d_counters_pad.data().get() + 2;
  d_left_unvisited_cnt = d_counters_pad.data().get() + 3;

  // Lets use this int* for the next 3 lines
  // Its dereferenced value is not initialized - so we dont care about what we
  // put in it
  IndexType *d_nisolated = d_new_frontier_cnt;
  cudaMemsetAsync(d_nisolated, 0, sizeof(IndexType), stream);

  // Computing isolated_bmap
  // Only dependent on graph - not source vertex - done once
  traversal::flag_isolated_vertices(number_of_vertices,
                                    isolated_bmap.data().get(),
                                    row_offsets,
                                    vertex_degree.data().get(),
                                    d_nisolated,
                                    stream);
  cudaMemcpyAsync(&nisolated, d_nisolated, sizeof(IndexType), cudaMemcpyDeviceToHost, stream);

  // We need nisolated to be ready to use
  cudaStreamSynchronize(stream);
}

template <typename IndexType>
void BFS<IndexType>::configure(IndexType *_distances,
                               IndexType *_predecessors)
{
  distances    = _distances;
  predecessors = _predecessors;

  computeDistances    = (distances != NULL);
  computePredecessors = (predecessors != NULL);

  // We need distances to use bottom up
  if (directed && !computeDistances) {
    distances_vals.resize(number_of_vertices);
    distances = distances_vals.data().get();
  }
}

template <typename IndexType>
void BFS<IndexType>::traverse(IndexType source_vertex)
{
  // Init visited_bmap
  // If the graph is undirected, we not that
  // we will never discover isolated vertices (in degree = out degree = 0)
  // we avoid a lot of work by flagging them now
  // in g500 graphs they represent ~25% of total vertices
  // more than that for wiki and twitter graphs

  if (directed) {
    cudaMemsetAsync(visited_bmap.data().get(), 0, vertices_bmap_size * sizeof(int), stream);
  } else {
    cudaMemcpyAsync(visited_bmap.data().get(),
                    isolated_bmap.data().get(),
                    vertices_bmap_size * sizeof(int),
                    cudaMemcpyDeviceToDevice,
                    stream);
  }

  // If needed, setting all vertices as undiscovered (inf distance)
  // We dont use computeDistances here
  // if the graph is undirected, we may need distances even if
  // computeDistances is false
  if (distances)
    traversal::fill_vec(distances, number_of_vertices, traversal::vec_t<IndexType>::max, stream);

  // If needed, setting all predecessors to non-existent (-1)
  if (computePredecessors) {
    cudaMemsetAsync(predecessors, -1, number_of_vertices * sizeof(IndexType), stream);
  }

  //
  // Initial frontier
  //

  frontier = original_frontier.data().get();

  if (distances) { cudaMemsetAsync(&distances[source_vertex], 0, sizeof(IndexType), stream); }

  // Setting source_vertex as visited
  // There may be bit already set on that bmap (isolated vertices) - if the
  // graph is undirected
  int current_visited_bmap_source_vert = 0;

  if (!directed) {
    cudaMemcpyAsync(&current_visited_bmap_source_vert,
                    visited_bmap.data().get() + (source_vertex / INT_SIZE),
                    sizeof(int),
                    cudaMemcpyDeviceToHost);
    // We need current_visited_bmap_source_vert
    cudaStreamSynchronize(stream);
  }

  int m = (1 << (source_vertex % INT_SIZE));

  // In that case, source is isolated, done now
  if (!directed && (m & current_visited_bmap_source_vert)) {
    // Init distances and predecessors are done, (cf Streamsync in previous if)
    return;
  }

  m |= current_visited_bmap_source_vert;

  cudaMemcpyAsync(visited_bmap.data().get() + (source_vertex / INT_SIZE),
                  &m,
                  sizeof(int),
                  cudaMemcpyHostToDevice,
                  stream);

  // Adding source_vertex to init frontier
  cudaMemcpyAsync(&frontier[0], &source_vertex, sizeof(IndexType), cudaMemcpyHostToDevice, stream);

  // mf : edges in frontier
  // nf : vertices in frontier
  // mu : edges undiscovered
  // nu : nodes undiscovered
  // lvl : current frontier's depth
  IndexType mf, nf, mu, nu;
  IndexType lvl = 1;

  // Frontier has one vertex
  nf = 1;

  // all edges are undiscovered (by def isolated vertices have 0 edges)
  mu = number_of_edges;

  // all non isolated vertices are undiscovered (excepted source vertex, which is in frontier)
  // That number is wrong if source_vertex is also isolated - but it's not important
  nu = number_of_vertices - nisolated - nf;

  IndexType size_last_left_unvisited_queue = number_of_vertices;  // we just need value > 0
  IndexType size_last_unvisited_queue      = 0;                   // queue empty

  // Typical pre-top down workflow. set_frontier_degree + exclusive-scan
  traversal::set_frontier_degree(
    frontier_vertex_degree, frontier, vertex_degree.data().get(), nf, stream);
  traversal::exclusive_sum(
    frontier_vertex_degree, exclusive_sum_frontier_vertex_degree, nf + 1, stream);

  cudaMemcpyAsync(&mf,
                  &exclusive_sum_frontier_vertex_degree[nf],
                  sizeof(IndexType),
                  cudaMemcpyDeviceToHost,
                  stream);

  // We need mf
  cudaStreamSynchronize(stream);

  // useDistances : we check if a vertex is a parent using distances in bottom up - distances become
  // working data undirected g : need parents to be in children's neighbors

  // In case the shortest path counters need to be computeed, the bottom_up approach cannot be used

  while (nf > 0) {
    // Each vertices can appear only once in the frontierer array - we know it will fit
    new_frontier     = frontier + nf;
    IndexType old_nf = nf;
    resetDevicePointers();

    // Executing algo
    traversal::compute_bucket_offsets(
      exclusive_sum_frontier_vertex_degree,
      exclusive_sum_frontier_vertex_buckets_offsets.data().get(),
      nf,
      mf,
      stream);
    bfs_kernels::frontier_expand(row_offsets,
                                  col_indices,
                                  frontier,
                                  nf,
                                  mf,
                                  lvl,
                                  new_frontier,
                                  d_new_frontier_cnt,
                                  exclusive_sum_frontier_vertex_degree,
                                  exclusive_sum_frontier_vertex_buckets_offsets.data().get(),
                                  visited_bmap.data().get(),
                                  distances,
                                  predecessors,
                                  isolated_bmap.data().get(),
                                  directed,
                                  stream);

    mu -= mf;

    cudaMemcpyAsync(&nf, d_new_frontier_cnt, sizeof(IndexType), cudaMemcpyDeviceToHost, stream);
    CHECK_CUDA(stream);

    // We need nf
    cudaStreamSynchronize(stream);

    if (nf) {
      // Typical pre-top down workflow. set_frontier_degree + exclusive-scan
      traversal::set_frontier_degree(
        frontier_vertex_degree, new_frontier, vertex_degree.data().get(), nf, stream);
      traversal::exclusive_sum(
        frontier_vertex_degree, exclusive_sum_frontier_vertex_degree, nf + 1, stream);
      cudaMemcpyAsync(&mf,
                      &exclusive_sum_frontier_vertex_degree[nf],
                      sizeof(IndexType),
                      cudaMemcpyDeviceToHost,
                      stream);

      // We need mf
      cudaStreamSynchronize(stream);
    }

    // Updating undiscovered edges count
    nu -= nf;

    // Using new frontier
    frontier = new_frontier;

    ++lvl;
  }
}

template <typename IndexType>
void BFS<IndexType>::resetDevicePointers()
{
  cudaMemsetAsync(d_counters_pad.data().get(), 0, 4 * sizeof(IndexType), stream);
}

template <typename IndexType>
void BFS<IndexType>::clean()
{
  // the vectors have a destructor that takes care of cleaning
}

// Explicit Instantiation
template class BFS<uint32_t>;
template class BFS<int>;
template class BFS<int64_t>;

}  // namespace detail

// NOTE: SP counter increase extremely fast on large graph
//       It can easily reach 1e40~1e70 on GAP-road.mtx
template <typename VT, typename ET, typename WT>
void bfs(raft::handle_t const &handle,
         GraphCSRView<VT, ET, WT> const &graph,
         VT *distances,
         VT *predecessors,
         double *sp_counters,
         const VT start_vertex,
         bool directed)
{
  static_assert(std::is_integral<VT>::value && sizeof(VT) >= sizeof(int32_t),
                "Unsupported vertex id data type. Use integral types of size >= sizeof(int32_t)");
  static_assert(std::is_same<VT, ET>::value,
                "VT and ET should be the same time for the current BFS implementation");
  static_assert(std::is_floating_point<WT>::value,
                "Unsupported edge weight type. Use floating point types");  // actually, this is
                                                                            // unnecessary for BFS
  if (handle.comms_initialized()) {
    CUGRAPH_EXPECTS(sp_counters == nullptr,
                    "BFS Traversal shortest path is not supported in OPG path");
    opg::bfs<VT, ET, WT>(handle, graph, distances, predecessors, start_vertex);
  } else {
    VT number_of_vertices = graph.number_of_vertices;
    ET number_of_edges    = graph.number_of_edges;

    const VT *indices_ptr = graph.indices;
    const ET *offsets_ptr = graph.offsets;

    int alpha = 15;
    int beta  = 18;
    // FIXME: Use VT and ET in the BFS detail
    cugraph::detail::BFS<VT> bfs(
      number_of_vertices, number_of_edges, offsets_ptr, indices_ptr, directed, alpha, beta);
    bfs.configure(distances, predecessors, sp_counters, nullptr);
    bfs.traverse(start_vertex);
  }
}

// Explicit Instantiation
template void bfs<uint32_t, uint32_t, float>(raft::handle_t const &handle,
                                             GraphCSRView<uint32_t, uint32_t, float> const &graph,
                                             uint32_t *distances,
                                             uint32_t *predecessors,
                                             double *sp_counters,
                                             const uint32_t source_vertex,
                                             bool directed);

// Explicit Instantiation
template void bfs<uint32_t, uint32_t, double>(raft::handle_t const &handle,
                                              GraphCSRView<uint32_t, uint32_t, double> const &graph,
                                              uint32_t *distances,
                                              uint32_t *predecessors,
                                              double *sp_counters,
                                              const uint32_t source_vertex,
                                              bool directed);

// Explicit Instantiation
template void bfs<int32_t, int32_t, float>(raft::handle_t const &handle,
                                           GraphCSRView<int32_t, int32_t, float> const &graph,
                                           int32_t *distances,
                                           int32_t *predecessors,
                                           double *sp_counters,
                                           const int32_t source_vertex,
                                           bool directed);

// Explicit Instantiation
template void bfs<int32_t, int32_t, double>(raft::handle_t const &handle,
                                            GraphCSRView<int32_t, int32_t, double> const &graph,
                                            int32_t *distances,
                                            int32_t *predecessors,
                                            double *sp_counters,
                                            const int32_t source_vertex,
                                            bool directed);

// Explicit Instantiation
template void bfs<int64_t, int64_t, float>(raft::handle_t const &handle,
                                           GraphCSRView<int64_t, int64_t, float> const &graph,
                                           int64_t *distances,
                                           int64_t *predecessors,
                                           double *sp_counters,
                                           const int64_t source_vertex,
                                           bool directed);

// Explicit Instantiation
template void bfs<int64_t, int64_t, double>(raft::handle_t const &handle,
                                            GraphCSRView<int64_t, int64_t, double> const &graph,
                                            int64_t *distances,
                                            int64_t *predecessors,
                                            double *sp_counters,
                                            const int64_t source_vertex,
                                            bool directed);

}  // namespace cugraph
