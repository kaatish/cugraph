/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <raft/integer_utils.h>

namespace cugraph {

namespace opg {

namespace detail {

template <typename degree_t>
constexpr int BitsPWrd = sizeof(degree_t) * 8;

template <typename degree_t>
constexpr int NumberBins = sizeof(degree_t) * 8 + 1;

const int BLOCK_SIZE_LB = 128;
const int EDGES_PER_THREAD = 8;

template <typename T>
constexpr inline T number_of_words(T number_of_bits)
{
  return raft::div_rounding_up_safe(number_of_bits, static_cast<T>(BitsPWrd<unsigned>));
}

struct bitwise_or {
  __device__ unsigned operator()(unsigned& a, unsigned& b) { return a | b; }
};

struct remove_visited {
  __device__ unsigned operator()(unsigned& visited, unsigned& output)
  {
    // OUTPUT AND VISITED - common bits between output and visited
    // OUTPUT AND (NOT (OUTPUT AND VISITED))
    // - remove common bits between output and visited from output
    return (output & (~(output & visited)));
  }
};

// Return true if the nth bit of an array is set to 1
template <typename T>
__device__ bool is_nth_bit_set(unsigned *bitmap, T index)
{
  return bitmap[index / BitsPWrd<unsigned>] & (unsigned{1} << (index % BitsPWrd<unsigned>));
}

struct check_frontier_candidate {
  unsigned *frontier_bmp_;
  unsigned *isolated_bmp_;
  unsigned *visited_bmp_;
  check_frontier_candidate(
      unsigned *frontier_bmp,
      unsigned *isolated_bmp,
      unsigned *visited_bmp) :
    frontier_bmp_(frontier_bmp),
    isolated_bmp_(isolated_bmp),
    visited_bmp_(visited_bmp) {}

  template <typename T>
  __device__
  bool operator()(const T id) {
    //True if vertex is in frontier but not isolated
    // and not visited
    return is_nth_bit_set(frontier_bmp_, id) &&
      !is_nth_bit_set(isolated_bmp_, id) &&
      !is_nth_bit_set(visited_bmp_, id);
  }
};

template <typename VT>
struct bfs_bmap_frontier_pred {
  unsigned* output_frontier_;
  unsigned* visited_;
  VT* predecessors_;

  bfs_bmap_frontier_pred(
      unsigned* output_frontier,
      unsigned* visited,
      VT* predecessors)
    : output_frontier_(output_frontier),
    visited_(visited),
    predecessors_(predecessors)
  {
  }

  __device__ void operator()(VT src, VT dst)
  {
    unsigned active_bit = static_cast<unsigned>(1) << (dst % BitsPWrd<unsigned>);
    unsigned prev_word  = atomicOr(output_frontier_ + (dst / BitsPWrd<unsigned>), active_bit);
    bool dst_not_visited_earlier = !(active_bit & visited_[dst / BitsPWrd<unsigned>]);
    bool dst_not_visited_current = !(prev_word & active_bit);
    // If this thread activates the frontier bitmap for a destination
    // then the source is the predecessor of that destination
    if (dst_not_visited_earlier && dst_not_visited_current) { predecessors_[dst] = src; }
  }
};

template <typename VT, typename ET>
struct bfs_pred {
  unsigned* output_frontier_;
  unsigned* isolated_;
  unsigned* visited_;
  VT* predecessors_;

  bfs_pred(
    unsigned* output_frontier, unsigned* isolated, unsigned* visited, VT* predecessors)
    : output_frontier_(output_frontier),
      isolated_(isolated),
      visited_(visited),
      predecessors_(predecessors)
  {
  }

  __device__ void operator()(VT src, VT dst, VT * frontier, ET * frontier_count)
  {
    printf("e : %d %d\n", (int)src, (int)dst);
    unsigned active_bit = static_cast<unsigned>(1) << (dst % BitsPWrd<unsigned>);
    unsigned prev_word  = atomicOr(output_frontier_ + (dst / BitsPWrd<unsigned>), active_bit);
    bool dst_not_visited_earlier = !(active_bit & visited_[dst / BitsPWrd<unsigned>]);
    bool dst_not_visited_current = !(prev_word & active_bit);
    // If this thread activates the frontier bitmap for a destination
    // then the source is the predecessor of that destination
    if (dst_not_visited_earlier && dst_not_visited_current) {
      predecessors_[dst] = src;
      if (!(active_bit & isolated_[dst / BitsPWrd<unsigned>])) {
        auto count = *frontier_count;
        frontier[count] = dst;
        *frontier_count = count+1;
      }
    }
  }
};

template <typename VT, typename ET>
struct bfs_pred_dist {
  unsigned* output_frontier_;
  unsigned* isolated_;
  unsigned* visited_;
  VT* predecessors_;
  VT* distances_;
  VT level_;

  bfs_pred_dist(
    unsigned* output_frontier, unsigned* isolated, unsigned* visited, VT* predecessors, VT* distances, VT level)
    : output_frontier_(output_frontier),
      isolated_(isolated),
      visited_(visited),
      predecessors_(predecessors),
      distances_(distances),
      level_(level)
  {
  }

  __device__ void operator()(VT src, VT dst, VT * frontier, ET * frontier_count)
  {
    printf("e : %d %d\n", (int)src, (int)dst);
    unsigned active_bit = static_cast<unsigned>(1) << (dst % BitsPWrd<unsigned>);
    unsigned prev_word  = atomicOr(output_frontier_ + (dst / BitsPWrd<unsigned>), active_bit);
    bool dst_not_visited_earlier = !(active_bit & visited_[dst / BitsPWrd<unsigned>]);
    bool dst_not_visited_current = !(prev_word & active_bit);
    // If this thread activates the frontier bitmap for a destination
    // then the source is the predecessor of that destination
    if (dst_not_visited_earlier && dst_not_visited_current) {
      distances_[dst]    = level_;
      predecessors_[dst] = src;
      if (!(active_bit & isolated_[dst / BitsPWrd<unsigned>])) {
        auto count = *frontier_count;
        frontier[count] = dst;
        *frontier_count = count+1;
      }
    }
  }
};

template <typename VT>
struct bfs_bmap_frontier_pred_dist {
  unsigned* output_frontier_;
  unsigned* visited_;
  VT* predecessors_;
  VT* distances_;
  VT level_;

  bfs_bmap_frontier_pred_dist(
    unsigned* output_frontier, unsigned* visited, VT* predecessors, VT* distances, VT level)
    : output_frontier_(output_frontier),
      visited_(visited),
      predecessors_(predecessors),
      distances_(distances),
      level_(level)
  {
  }

  __device__ void operator()(VT src, VT dst)
  {
    unsigned active_bit = static_cast<unsigned>(1) << (dst % BitsPWrd<unsigned>);
    unsigned prev_word  = atomicOr(output_frontier_ + (dst / BitsPWrd<unsigned>), active_bit);
    bool dst_not_visited_earlier = !(active_bit & visited_[dst / BitsPWrd<unsigned>]);
    bool dst_not_visited_current = !(prev_word & active_bit);
    // If this thread activates the frontier bitmap for a destination
    // then the source is the predecessor of that destination
    if (dst_not_visited_earlier && dst_not_visited_current) {
      distances_[dst]    = level_;
      predecessors_[dst] = src;
    }
  }
};

struct is_not_equal {
  unsigned cmp_;
  unsigned* flag_;
  is_not_equal(unsigned cmp, unsigned* flag) : cmp_(cmp), flag_(flag) {}
  __device__ void operator()(unsigned& val)
  {
    if (val != cmp_) { *flag_ = 1; }
  }
};

}  // namespace detail

}  // namespace opg

}  // namespace cugraph
