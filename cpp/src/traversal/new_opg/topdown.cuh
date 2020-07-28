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

#include "../traversal_common.cuh"

namespace cugraph {
namespace opg {
namespace detail {

template <typename vertex_t, typename edge_t>
__global__ void topdown_expand_kernel(
  const edge_t *offsets,
  const vertex_t *indices,
  const vertex_t *frontier,
  const vertex_t frontier_size,
  const unsigned *isolated_bmap,
  const vertex_t level,
  const edge_t totaldegree,
  const edge_t max_items_per_thread,
  const edge_t *frontier_degrees_exclusive_sum,
  const edge_t *frontier_degrees_exclusive_sum_buckets_offsets,
  int *bmap,
  vertex_t * new_frontier,
  vertex_t new_frontier_cnt,
  vertex_t *distances,
  vertex_t *predecessors)
{
}
//
//
//
//  const IndexType totaldegree,
//  const IndexType max_items_per_thread,
//  const IndexType lvl,
//  IndexType *new_frontier,
//  IndexType *new_frontier_cnt,
//  const IndexType *frontier_degrees_exclusive_sum,
//  const IndexType *frontier_degrees_exclusive_sum_buckets_offsets,
//  int *bmap,
//  IndexType *distances,
//  IndexType *predecessors,
//  const int *isolated_bmap)
//{
//  // BlockScan
//  typedef cub::BlockScan<IndexType, TOP_DOWN_EXPAND_DIMX> BlockScan;
//  __shared__ typename BlockScan::TempStorage scan_storage;
//
//  // We will do a scan to know where to write in frontier
//  // This will contain the common offset of the block
//  __shared__ IndexType frontier_common_block_offset;
//
//  __shared__ IndexType shared_buckets_offsets[TOP_DOWN_EXPAND_DIMX - NBUCKETS_PER_BLOCK + 1];
//  __shared__ IndexType shared_frontier_degrees_exclusive_sum[TOP_DOWN_EXPAND_DIMX + 1];
//
//  //
//  // Frontier candidates local queue
//  // We process TOP_DOWN_BATCH_SIZE vertices in parallel, so we need to be able
//  // to store everything We also save the predecessors here, because we will not
//  // be able to retrieve it after
//  //
//  __shared__ IndexType
//    shared_local_new_frontier_candidates[TOP_DOWN_BATCH_SIZE * TOP_DOWN_EXPAND_DIMX];
//  __shared__ IndexType
//    shared_local_new_frontier_predecessors[TOP_DOWN_BATCH_SIZE * TOP_DOWN_EXPAND_DIMX];
//  __shared__ IndexType block_n_frontier_candidates;
//
//  IndexType block_offset = (blockDim.x * blockIdx.x) * max_items_per_thread;
//
//  // When this kernel is converted to support different VT and ET, this
//  // will likely split into invalid_vid and invalid_eid
//  // This is equivalent to ~IndexType(0) (i.e., all bits set to 1)
//  constexpr IndexType invalid_idx = cugraph::invalid_idx<IndexType>::value;
//
//  IndexType n_items_per_thread_left =
//    (totaldegree > block_offset)
//      ? (totaldegree - block_offset + TOP_DOWN_EXPAND_DIMX - 1) / TOP_DOWN_EXPAND_DIMX
//      : 0;
//
//  n_items_per_thread_left = min(max_items_per_thread, n_items_per_thread_left);
//
//  for (; (n_items_per_thread_left > 0) && (block_offset < totaldegree);
//
//       block_offset += MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD * blockDim.x,
//       n_items_per_thread_left -= min(
//         n_items_per_thread_left, static_cast<IndexType>(MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD))) {
//    // In this loop, we will process batch_set_size batches
//    IndexType nitems_per_thread =
//      min(n_items_per_thread_left, static_cast<IndexType>(MAX_ITEMS_PER_THREAD_PER_OFFSETS_LOAD));
//
//    // Loading buckets offset (see compute_bucket_offsets_kernel)
//
//    if (threadIdx.x < (nitems_per_thread * NBUCKETS_PER_BLOCK + 1))
//      shared_buckets_offsets[threadIdx.x] =
//        frontier_degrees_exclusive_sum_buckets_offsets[block_offset / TOP_DOWN_BUCKET_SIZE +
//                                                       threadIdx.x];
//
//    // We will use shared_buckets_offsets
//    __syncthreads();
//
//    //
//    // shared_buckets_offsets gives us a range of the possible indexes
//    // for edge of linear_threadx, we are looking for the value k such as
//    // k is the max value such as frontier_degrees_exclusive_sum[k] <=
//    // linear_threadx
//    //
//    // we have 0 <= k < frontier_size
//    // but we also have :
//    //
//    // frontier_degrees_exclusive_sum_buckets_offsets[linear_threadx/TOP_DOWN_BUCKET_SIZE]
//    // <= k
//    // <=
//    // frontier_degrees_exclusive_sum_buckets_offsets[linear_threadx/TOP_DOWN_BUCKET_SIZE
//    // + 1]
//    //
//    // To find the exact value in that range, we need a few values from
//    // frontier_degrees_exclusive_sum (see below) We will load them here We will
//    // load as much as we can - if it doesn't fit we will make multiple
//    // iteration of the next loop Because all vertices in frontier have degree >
//    // 0, we know it will fits if left + 1 = right (see below)
//
//    // We're going to load values in frontier_degrees_exclusive_sum for batch
//    // [left; right[ If it doesn't fit, --right until it does, then loop It is
//    // excepted to fit on the first try, that's why we start right =
//    // nitems_per_thread
//
//    IndexType left  = 0;
//    IndexType right = nitems_per_thread;
//
//    while (left < nitems_per_thread) {
//      //
//      // Values that are necessary to compute the local binary searches
//      // We only need those with indexes between extremes indexes of
//      // buckets_offsets We need the next val for the binary search, hence the
//      // +1
//      //
//
//      IndexType nvalues_to_load = shared_buckets_offsets[right * NBUCKETS_PER_BLOCK] -
//                                  shared_buckets_offsets[left * NBUCKETS_PER_BLOCK] + 1;
//
//      // If left = right + 1 we are sure to have nvalues_to_load <
//      // TOP_DOWN_EXPAND_DIMX+1
//      while (nvalues_to_load > (TOP_DOWN_EXPAND_DIMX + 1)) {
//        --right;
//
//        nvalues_to_load = shared_buckets_offsets[right * NBUCKETS_PER_BLOCK] -
//                          shared_buckets_offsets[left * NBUCKETS_PER_BLOCK] + 1;
//      }
//
//      IndexType nitems_per_thread_for_this_load = right - left;
//
//      IndexType frontier_degrees_exclusive_sum_block_offset =
//        shared_buckets_offsets[left * NBUCKETS_PER_BLOCK];
//
//      if (threadIdx.x < nvalues_to_load) {
//        shared_frontier_degrees_exclusive_sum[threadIdx.x] =
//          frontier_degrees_exclusive_sum[frontier_degrees_exclusive_sum_block_offset + threadIdx.x];
//      }
//
//      if (nvalues_to_load == (TOP_DOWN_EXPAND_DIMX + 1) && threadIdx.x == 0) {
//        shared_frontier_degrees_exclusive_sum[TOP_DOWN_EXPAND_DIMX] =
//          frontier_degrees_exclusive_sum[frontier_degrees_exclusive_sum_block_offset +
//                                         TOP_DOWN_EXPAND_DIMX];
//      }
//
//      // shared_frontier_degrees_exclusive_sum is in shared mem, we will use it,
//      // sync
//      __syncthreads();
//
//      // Now we will process the edges
//      // Here each thread will process nitems_per_thread_for_this_load
//      for (IndexType item_index = 0; item_index < nitems_per_thread_for_this_load;
//           item_index += TOP_DOWN_BATCH_SIZE) {
//        // We process TOP_DOWN_BATCH_SIZE edge in parallel (instruction
//        // parallism) Reduces latency
//
//        IndexType current_max_edge_index = min(
//          static_cast<size_t>(block_offset) + (left + nitems_per_thread_for_this_load) * blockDim.x,
//          static_cast<size_t>(totaldegree));
//
//        // We will need vec_u (source of the edge) until the end if we need to
//        // save the predecessors For others informations, we will reuse pointers
//        // on the go (nvcc does not color well the registers in that case)
//
//        IndexType vec_u[TOP_DOWN_BATCH_SIZE];
//        IndexType local_buf1[TOP_DOWN_BATCH_SIZE];
//        IndexType local_buf2[TOP_DOWN_BATCH_SIZE];
//
//        IndexType *vec_frontier_degrees_exclusive_sum_index = &local_buf2[0];
//
//#pragma unroll
//        for (IndexType iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
//          IndexType ibatch = left + item_index + iv;
//          IndexType gid    = block_offset + ibatch * blockDim.x + threadIdx.x;
//
//          if (gid < current_max_edge_index) {
//            IndexType start_off_idx = (ibatch * blockDim.x + threadIdx.x) / TOP_DOWN_BUCKET_SIZE;
//            IndexType bucket_start =
//              shared_buckets_offsets[start_off_idx] - frontier_degrees_exclusive_sum_block_offset;
//            IndexType bucket_end = shared_buckets_offsets[start_off_idx + 1] -
//                                   frontier_degrees_exclusive_sum_block_offset;
//
//            IndexType k = traversal::binsearch_maxle(
//                            shared_frontier_degrees_exclusive_sum, gid, bucket_start, bucket_end) +
//                          frontier_degrees_exclusive_sum_block_offset;
//            vec_u[iv]                                    = frontier[k];  // origin of this edge
//            vec_frontier_degrees_exclusive_sum_index[iv] = frontier_degrees_exclusive_sum[k];
//          } else {
//            vec_u[iv]                                    = invalid_idx;
//            vec_frontier_degrees_exclusive_sum_index[iv] = invalid_idx;
//          }
//        }
//
//        IndexType *vec_row_ptr_u = &local_buf1[0];
//#pragma unroll
//        for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
//          IndexType u = vec_u[iv];
//          // row_ptr for this vertex origin u
//          vec_row_ptr_u[iv] = (u != invalid_idx) ? row_ptr[u] : invalid_idx;
//        }
//
//        // We won't need row_ptr after that, reusing pointer
//        IndexType *vec_dest_v = vec_row_ptr_u;
//
//#pragma unroll
//        for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
//          IndexType thread_item_index = left + item_index + iv;
//          IndexType gid               = block_offset + thread_item_index * blockDim.x + threadIdx.x;
//
//          IndexType row_ptr_u = vec_row_ptr_u[iv];
//          // Need this check so that we don't use invalid values of edge to index
//          if (row_ptr_u != invalid_idx) {
//            IndexType edge = row_ptr_u + gid - vec_frontier_degrees_exclusive_sum_index[iv];
//
//            // Destination of this edge
//            vec_dest_v[iv] = col_ind[edge];
//          }
//        }
//
//        // We don't need vec_frontier_degrees_exclusive_sum_index anymore
//        IndexType *vec_v_visited_bmap = vec_frontier_degrees_exclusive_sum_index;
//
//        // Visited bmap need to contain information about the previous
//        // frontier if we actually process every edge (shortest path counting)
//        // otherwise we can read and update from the same bmap
//#pragma unroll
//        for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
//          IndexType v = vec_dest_v[iv];
//          vec_v_visited_bmap[iv] =
//            (v != invalid_idx) ? bmap[v / INT_SIZE] : (~int(0));  // will look visited
//        }
//
//        // From now on we will consider v as a frontier candidate
//        // If for some reason vec_candidate[iv] should be put in the
//        // new_frontier Then set vec_candidate[iv] = -1
//        IndexType *vec_frontier_candidate = vec_dest_v;
//
//#pragma unroll
//        for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
//          IndexType v = vec_frontier_candidate[iv];
//          int m       = 1 << (v % INT_SIZE);
//
//          int is_visited = vec_v_visited_bmap[iv] & m;
//
//          if (is_visited) vec_frontier_candidate[iv] = invalid_idx;
//        }
//
//        // vec_v_visited_bmap is available
//        IndexType *vec_is_isolated_bmap = vec_v_visited_bmap;
//
//#pragma unroll
//        for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
//          IndexType v              = vec_frontier_candidate[iv];
//          vec_is_isolated_bmap[iv] = (v != invalid_idx) ? isolated_bmap[v / INT_SIZE] : ~int(0);
//        }
//
//#pragma unroll
//        for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
//          IndexType v     = vec_frontier_candidate[iv];
//          int m           = 1 << (v % INT_SIZE);
//          int is_isolated = vec_is_isolated_bmap[iv] & m;
//
//          // If v is isolated, we will not add it to the frontier (it's not a
//          // frontier candidate) 1st reason : it's useless 2nd reason : it
//          // will make top down algo fail we need each node in frontier to
//          // have a degree > 0 If it is isolated, we just need to mark it as
//          // visited, and save distance and predecessor here. Not need to
//          // check return value of atomicOr
//
//          if (is_isolated && v != invalid_idx) {
//            int m = 1 << (v % INT_SIZE);
//            atomicOr(&bmap[v / INT_SIZE], m);
//            if (distances) distances[v] = lvl;
//
//            if (predecessors) predecessors[v] = vec_u[iv];
//
//            // This is no longer a candidate, neutralize it
//            vec_frontier_candidate[iv] = invalid_idx;
//          }
//        }
//
//        // Number of successor candidate hold by this thread
//        IndexType thread_n_frontier_candidates = 0;
//
//#pragma unroll
//        for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
//          IndexType v = vec_frontier_candidate[iv];
//          if (v != invalid_idx) ++thread_n_frontier_candidates;
//        }
//
//        // We need to have all nfrontier_candidates to be ready before doing the
//        // scan
//        __syncthreads();
//
//        // We will put the frontier candidates in a local queue
//        // Computing offsets
//        IndexType thread_frontier_candidate_offset = 0;  // offset inside block
//        BlockScan(scan_storage)
//          .ExclusiveSum(thread_n_frontier_candidates, thread_frontier_candidate_offset);
//
//#pragma unroll
//        for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
//          // May have bank conflicts
//          IndexType frontier_candidate = vec_frontier_candidate[iv];
//
//          if (frontier_candidate != invalid_idx) {
//            shared_local_new_frontier_candidates[thread_frontier_candidate_offset] =
//              frontier_candidate;
//            shared_local_new_frontier_predecessors[thread_frontier_candidate_offset] = vec_u[iv];
//            ++thread_frontier_candidate_offset;
//          }
//        }
//
//        if (threadIdx.x == (TOP_DOWN_EXPAND_DIMX - 1)) {
//          // No need to add nsuccessor_candidate, even if its an
//          // exclusive sum
//          // We incremented the thread_frontier_candidate_offset
//          block_n_frontier_candidates = thread_frontier_candidate_offset;
//        }
//
//        // broadcast block_n_frontier_candidates
//        __syncthreads();
//
//        IndexType naccepted_vertices = 0;
//        // We won't need vec_frontier_candidate after that
//        IndexType *vec_frontier_accepted_vertex = vec_frontier_candidate;
//
//#pragma unroll
//        for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
//          const int idx_shared             = iv * blockDim.x + threadIdx.x;
//          vec_frontier_accepted_vertex[iv] = invalid_idx;
//
//          if (idx_shared < block_n_frontier_candidates) {
//            IndexType v = shared_local_new_frontier_candidates[idx_shared];  // popping
//                                                                             // queue
//            int m = 1 << (v % INT_SIZE);
//            int q = atomicOr(&bmap[v / INT_SIZE], m);  // atomicOr returns old
//
//            if (!(m & q)) {  // if this thread was the first to discover this node
//              if (distances) distances[v] = lvl;
//
//              if (predecessors) {
//                IndexType pred  = shared_local_new_frontier_predecessors[idx_shared];
//                predecessors[v] = pred;
//              }
//
//              vec_frontier_accepted_vertex[iv] = v;
//              ++naccepted_vertices;
//            }
//          }
//        }
//
//        // We need naccepted_vertices to be ready
//        __syncthreads();
//
//        IndexType thread_new_frontier_offset;
//
//        BlockScan(scan_storage).ExclusiveSum(naccepted_vertices, thread_new_frontier_offset);
//
//        if (threadIdx.x == (TOP_DOWN_EXPAND_DIMX - 1)) {
//          IndexType inclusive_sum = thread_new_frontier_offset + naccepted_vertices;
//          // for this thread, thread_new_frontier_offset + has_successor
//          // (exclusive sum)
//          if (inclusive_sum)
//            frontier_common_block_offset = traversal::atomicAdd(new_frontier_cnt, inclusive_sum);
//        }
//
//        // Broadcasting frontier_common_block_offset
//        __syncthreads();
//
//#pragma unroll
//        for (int iv = 0; iv < TOP_DOWN_BATCH_SIZE; ++iv) {
//          const int idx_shared = iv * blockDim.x + threadIdx.x;
//          if (idx_shared < block_n_frontier_candidates) {
//            IndexType new_frontier_vertex = vec_frontier_accepted_vertex[iv];
//
//            if (new_frontier_vertex != invalid_idx) {
//              IndexType off     = frontier_common_block_offset + thread_new_frontier_offset++;
//              new_frontier[off] = new_frontier_vertex;
//            }
//          }
//        }
//      }
//
//      // We need to keep shared_frontier_degrees_exclusive_sum coherent
//      __syncthreads();
//
//      // Preparing for next load
//      left  = right;
//      right = nitems_per_thread;
//    }
//
//    // we need to keep shared_buckets_offsets coherent
//    __syncthreads();
//  }
//}

}  // namespace detail
}  // namespace opg
}  // namespace cugraph
