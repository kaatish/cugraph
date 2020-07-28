/*
 * Copyright (c) 2018-2020 NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <iostream>

#include <utilities/sm_utils.h>
#include <cub/cub.cuh>

#include "graph.hpp"
#include "../traversal_common.cuh"
#include "topdown.cuh"

namespace cugraph {
namespace detail {
namespace bfs_kernels {

// topdown_expand_kernel
// Read current frontier and compute new one with top down paradigm
// One thread = One edge
// To know origin of edge, we have to find where is index_edge in the values of
// frontier_degrees_exclusive_sum (using a binary search, max less or equal
// than) This index k will give us the origin of this edge, which is frontier[k]
// This thread will then process the (linear_idx_thread -
// frontier_degrees_exclusive_sum[k])-ith edge of vertex frontier[k]
//
// To process blockDim.x = TOP_DOWN_EXPAND_DIMX edges, we need to first load
// NBUCKETS_PER_BLOCK bucket offsets - those will help us do the binary searches
// We can load up to TOP_DOWN_EXPAND_DIMX of those bucket offsets - that way we
// prepare for the next MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD
// * blockDim.x edges
//
// Once we have those offsets, we may still need a few values from
// frontier_degrees_exclusive_sum to compute exact index k To be able to do it,
// we will load the values that we need from frontier_degrees_exclusive_sum in
// shared memory We know that it will fit because we never add node with degree
// == 0 in the frontier, so we have an upper bound on the number of value to
// load (see below)
//
// We will then look which vertices are not visited yet :
// 1) if the unvisited vertex is isolated (=> degree == 0), we mark it as
// visited, update distances and predecessors, and move on 2) if the unvisited
// vertex has degree > 0, we add it to the "frontier_candidates" queue
//
// We then treat the candidates queue using the threadIdx.x < ncandidates
// If we are indeed the first thread to discover that vertex (result of
// atomicOr(visited)) We add it to the new frontier
//

template <typename IndexType>
void frontier_expand(const IndexType *row_ptr,
                     const IndexType *col_ind,
                     const IndexType *frontier,
                     const IndexType frontier_size,
                     const IndexType totaldegree,
                     const IndexType lvl,
                     IndexType *new_frontier,
                     IndexType *new_frontier_cnt,
                     const IndexType *frontier_degrees_exclusive_sum,
                     const IndexType *frontier_degrees_exclusive_sum_buckets_offsets,
                     int *visited_bmap,
                     IndexType *distances,
                     IndexType *predecessors,
                     const int *isolated_bmap,
                     bool directed,
                     cudaStream_t m_stream)
{
  if (!totaldegree) return;

  dim3 block;
  block.x = TOP_DOWN_EXPAND_DIMX;

  IndexType max_items_per_thread =
    (static_cast<size_t>(totaldegree) + MAXBLOCKS * block.x - 1) / (MAXBLOCKS * block.x);

  dim3 grid;
  grid.x = std::min((static_cast<size_t>(totaldegree) + max_items_per_thread * block.x - 1) /
                      (max_items_per_thread * block.x),
                    static_cast<size_t>(MAXBLOCKS));

  topdown_expand_kernel<<<grid, block, 0, m_stream>>>(
    row_ptr,
    col_ind,
    frontier,
    frontier_size,
    totaldegree,
    max_items_per_thread,
    lvl,
    new_frontier,
    new_frontier_cnt,
    frontier_degrees_exclusive_sum,
    frontier_degrees_exclusive_sum_buckets_offsets,
    visited_bmap,
    distances,
    predecessors,
    isolated_bmap,
    directed);
  CHECK_CUDA(m_stream);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void frontier_expand(
  cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
  rmm::device_vector<vertex_t> &input_frontier,
  vertex_t vertex_begin_index,
  vertex_t vertex_end_index,
  edge_t total_frontier_edge_count,
  rmm::device_vector<edge_t> &frontier_vertex_offset_,
  rmm::device_vector<edge_t> &frontier_vertex_bucket_offset_,
  rmm::device_vector<unsigned> &visited_bmap,
  rmm::device_vector<unsigned> &isolated_bmap,
  rmm::device_vector<vertex_t> &output_frontier,
  rmm::device_vector<edge_t> &output_frontier_size,
  vertex_t *distances,
  vertex_t *predecessors,
  cudaStream_t stream) {
}

}  // namespace bfs_kernels
}  // namespace detail
}  // namespace cugraph
