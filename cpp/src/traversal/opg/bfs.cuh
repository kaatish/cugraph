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

#include <raft/handle.hpp>
#include "bfs_comms.cuh"
#include "common_utils.cuh"
#include "load_balance.cuh"
#include "frontier_expand.cuh"
#include <string>
#include <thrust/scan.h>
#include <utilities/high_res_timer.hpp>

namespace cugraph {

namespace opg {

namespace detail {

template <typename edge_t>
struct isDegreeZero {
edge_t * offset_;
isDegreeZero(edge_t * offset) : offset_(offset) {}

__device__
bool operator()(const edge_t& id) {
  return (offset_[id+1] == offset_[id]);
}
};

struct set_nth_bit {
unsigned * bmap_;
set_nth_bit(unsigned * bmap) : bmap_(bmap) {}

template <typename T>
__device__
void operator()(const T& id) {
  atomicOr(
      bmap_ + (id / BitsPWrd<unsigned>),
      (unsigned{1} << (id % BitsPWrd<unsigned>)));
}

};

template <typename T>
void add_to_bitmap(raft::handle_t const &handle,
    rmm::device_vector<unsigned> &bmap,
    rmm::device_vector<T> &id) {
  cudaStream_t stream = handle.get_stream();
  thrust::for_each(rmm::exec_policy(stream)->on(stream),
      id.begin(), id.end(),
      set_nth_bit(bmap.data().get()));
}

template <typename T>
void collect_vectors(raft::handle_t const &handle,
    rmm::device_vector<size_t> &buffer_len,
    rmm::device_vector<T> &local,
    rmm::device_vector<T> &global) {
  auto my_rank = handle.get_comms().get_rank();
  buffer_len[my_rank] = local.size();
  handle.get_comms().allgather(
      buffer_len.data().get() + my_rank, buffer_len.data().get(),
      1, handle.get_stream());
  //buffer_len now contains the lengths of all local buffers
  //for all ranks

  thrust::host_vector<size_t> h_buffer_len = buffer_len;
  //h_buffer_offsets has to be int because raft allgatherv expects
  //int array for displacement vector. This should be changed in
  //raft so that the displacement is templated
  thrust::host_vector<int> h_buffer_offsets(h_buffer_len.size());
  int global_buffer_len = 0;
  for (size_t i = 0; i < h_buffer_offsets.size(); ++i) {
    h_buffer_offsets[i] = global_buffer_len;
    global_buffer_len += h_buffer_len[i];
  }
  global.resize(global_buffer_len);

  handle.get_comms().allgatherv(
      local.data().get(),
      global.data().get(),
      h_buffer_len.data(),
      h_buffer_offsets.data(),
      handle.get_stream());
}

template <typename vertex_t, typename edge_t, typename weight_t>
void bitmap_to_frontier(raft::handle_t const &handle,
    cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
    rmm::device_vector<unsigned> &output_frontier_bitmap,
    rmm::device_vector<unsigned> &isolated_bmap,
    rmm::device_vector<unsigned> &visited_bmap,
    rmm::device_vector<vertex_t> &vertex_ids) {

  cudaStream_t stream = handle.get_stream();
  vertex_ids.resize(graph.number_of_vertices);

  auto count = thrust::copy_if(rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator<vertex_t>(0),
      thrust::make_counting_iterator<vertex_t>(graph.number_of_vertices),
      thrust::make_counting_iterator<edge_t>(0),
      vertex_ids.begin(),
      check_frontier_candidate(
        output_frontier_bitmap.data().get(),
        isolated_bmap.data().get(),
        visited_bmap.data().get())) -
    vertex_ids.begin();

  vertex_ids.resize(count);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void populate_isolated_vertices(raft::handle_t const &handle,
    cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
    rmm::device_vector<vertex_t> &isolated_vertex_ids) {

  bool is_opg = (handle.comms_initialized() && (graph.local_vertices != nullptr) &&
                  (graph.local_offsets != nullptr));
  cudaStream_t stream = handle.get_stream();

  edge_t vertex_begin_, vertex_end_;
  if (is_opg) {
    isolated_vertex_ids.resize(graph.local_vertices[handle.get_comms().get_rank()]);
    vertex_begin_ = graph.local_offsets[handle.get_comms().get_rank()];
    vertex_end_   = graph.local_offsets[handle.get_comms().get_rank()] +
                  graph.local_vertices[handle.get_comms().get_rank()];
  } else {
    isolated_vertex_ids.resize(graph.number_of_vertices);
    vertex_begin_ = 0;
    vertex_end_   = graph.number_of_vertices;
  }
  auto count = thrust::copy_if(rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator<vertex_t>(vertex_begin_),
      thrust::make_counting_iterator<vertex_t>(vertex_end_),
      thrust::make_counting_iterator<edge_t>(0),
      isolated_vertex_ids.begin(),
      isDegreeZero<edge_t>(graph.offsets)) - isolated_vertex_ids.begin();
  isolated_vertex_ids.resize(count);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void create_isolated_bitmap(raft::handle_t const &handle,
    cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
    rmm::device_vector<vertex_t> &local_isolated_ids,
    rmm::device_vector<vertex_t> &global_isolated_ids,
    rmm::device_vector<size_t> &temp_buffer_len,
    rmm::device_vector<unsigned> &isolated_bmap) {

  size_t word_count = detail::number_of_words(graph.number_of_vertices);
  local_isolated_ids.resize(graph.number_of_vertices);
  global_isolated_ids.resize(graph.number_of_vertices);
  temp_buffer_len.resize(handle.get_comms().get_size());
  isolated_bmap.resize(word_count);

  populate_isolated_vertices(handle, graph, local_isolated_ids);
  collect_vectors(
      handle,
      temp_buffer_len,
      local_isolated_ids,
      global_isolated_ids);
  add_to_bitmap(handle, isolated_bmap, global_isolated_ids);
}

template <typename T>
void remove_duplicates(raft::handle_t const &handle,
    rmm::device_vector<T> &data)
{
  cudaStream_t stream = handle.get_stream();
  thrust::sort(
      rmm::exec_policy(stream)->on(stream),
      data.begin(), data.end());
  auto unique_count = thrust::unique(
      rmm::exec_policy(stream)->on(stream),
      data.begin(), data.end()) - data.begin();
  data.resize(unique_count);
}

} //namespace detail

#if 1
template <typename vertex_t, typename edge_t, typename weight_t>
void bfs(raft::handle_t const &handle,
         cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
         vertex_t *distances,
         vertex_t *predecessors,
         const vertex_t start_vertex)
{
  CUGRAPH_EXPECTS(handle.comms_initialized(),
                  "cugraph::mg::bfs() expected to work only in multi gpu case.");

  size_t word_count = detail::number_of_words(graph.number_of_vertices);
  rmm::device_vector<unsigned> isolated_bmap(word_count, 0);
  rmm::device_vector<unsigned> visited_bmap(word_count, 0);
  rmm::device_vector<unsigned> output_frontier_bmap(word_count, 0);

  //Buffers required for BFS
  rmm::device_vector<vertex_t> input_frontier(graph.number_of_vertices);
  rmm::device_vector<vertex_t> output_frontier(graph.number_of_vertices);
  rmm::device_vector<size_t> temp_buffer_len(handle.get_comms().get_size());

  // Load balancer for calls to bfs functors
  detail::LoadBalanceExecution<vertex_t, edge_t, weight_t> lb(handle, graph);

  HighResTimer main_loop_timer;
  HighResTimer step_timer;

  //Reusing buffers to create isolated bitmap
  {
    rmm::device_vector<vertex_t>& local_isolated_ids = input_frontier;
    rmm::device_vector<vertex_t>& global_isolated_ids = output_frontier;
    detail::create_isolated_bitmap(
        handle, graph,
        local_isolated_ids, global_isolated_ids,
        temp_buffer_len, isolated_bmap);
  }

  cudaStream_t stream = handle.get_stream();

  //Initialize input frontier
  input_frontier.resize(1);
  input_frontier[0] = start_vertex;

  //Start at level 0
  vertex_t level = 0;
  if (distances != nullptr) {
    thrust::fill(rmm::exec_policy(stream)->on(stream),
                 distances,
                 distances + graph.number_of_vertices,
                 std::numeric_limits<vertex_t>::max());
    thrust::device_ptr<vertex_t> dist(distances);
    dist[start_vertex] = vertex_t{0};
  }

  // Fill predecessors with graph.number_of_vertices
  // This will later be replaced by invalid_vertex_id
  thrust::fill(rmm::exec_policy(stream)->on(stream),
               predecessors,
               predecessors + graph.number_of_vertices,
               graph.number_of_vertices);

  do {
    main_loop_timer.start("main_loop");
    step_timer.start("add_to_bitmap 1");
    //Mark all input frontier vertices as visited
    detail::add_to_bitmap(handle, visited_bmap, input_frontier);
    step_timer.stop();

    step_timer.start("fill 2");
    //Clear output frontier bitmap
    thrust::fill(rmm::exec_policy(stream)->on(stream),
                 output_frontier_bmap.begin(),
                 output_frontier_bmap.end(),
                 static_cast<unsigned>(0));
    step_timer.stop();

    step_timer.start("lb.run 3");
    //Generate output frontier bitmap from input frontier
    if (distances != nullptr) {
      // BFS Functor for frontier calculation
      detail::bfs_bmap_frontier_pred_dist<vertex_t> bfs_op(
        output_frontier_bmap.data().get(),
        visited_bmap.data().get(),
        predecessors, distances, ++level);
      lb.run(bfs_op, input_frontier);
    } else {
      // BFS Functor for frontier calculation
      detail::bfs_bmap_frontier_pred<vertex_t> bfs_op(
        output_frontier_bmap.data().get(),
        visited_bmap.data().get(),
        predecessors);
      lb.run(bfs_op, input_frontier);
    }
    step_timer.stop();
    step_timer.start("bitmap_to_frontier 4");
    detail::bitmap_to_frontier(
        handle, graph,
        output_frontier_bmap,
        isolated_bmap,
        visited_bmap,
        output_frontier);
    step_timer.stop();

    step_timer.start("collect");
    //Use input_frontier buffer to collect output_frontier
    //from all the GPUs
    detail::collect_vectors(
        handle,
        temp_buffer_len,
        output_frontier,
        input_frontier);
    step_timer.stop();

    step_timer.start("remove duplicates");
    //Remove duplicates from input_frontier
    detail::remove_duplicates(handle, input_frontier);
    step_timer.stop();

    main_loop_timer.stop();
  } while (input_frontier.size() != 0);

  main_loop_timer.display(std::cout);
  step_timer.display(std::cout);
  // In place reduce to collect predecessors
  if (handle.comms_initialized()) {
    handle.get_comms().allreduce(predecessors,
                                 predecessors,
                                 graph.number_of_vertices,
                                 raft::comms::op_t::MIN,
                                 handle.get_stream());
  }

  // If the bfs loop does not assign a predecessor for a vertex
  // then its value will be graph.number_of_vertices. This needs to be
  // replaced by invalid vertex id to denote that a vertex does have
  // a predecessor
  thrust::replace(rmm::exec_policy(stream)->on(stream),
                  predecessors,
                  predecessors + graph.number_of_vertices,
                  graph.number_of_vertices,
                  cugraph::invalid_vertex_id<vertex_t>::value);

  if (distances != nullptr) {
    // In place reduce to collect predecessors
    if (handle.comms_initialized()) {
      handle.get_comms().allreduce(distances,
                                   distances,
                                   graph.number_of_vertices,
                                   raft::comms::op_t::MIN,
                                   handle.get_stream());
    }
  }

}

#else

template <typename VT, typename ET, typename WT>
void bfs(raft::handle_t const &handle,
         cugraph::GraphCSRView<VT, ET, WT> const &graph,
         VT *distances,
         VT *predecessors,
         const VT start_vertex)
{
  using namespace detail;

  // We need to keep track if a vertex is visited or its status
  // This needs to be done for all the vertices in the global graph
  size_t word_count = detail::number_of_words(graph.number_of_vertices);
  rmm::device_vector<unsigned> input_frontier(word_count);
  rmm::device_vector<unsigned> output_frontier(word_count);
  rmm::device_vector<unsigned> visited(word_count);

  rmm::device_vector<unsigned> frontier_not_empty(1);

  // Load balancer for calls to bfs functors
  LoadBalanceExecution<VT, ET, WT> lb(handle, graph);

  // Functor to check if frontier is empty
  is_not_equal neq(static_cast<unsigned>(0), frontier_not_empty.data().get());

  cudaStream_t stream = handle.get_stream();

  // Fill predecessors with an invalid vertex id
  thrust::fill(rmm::exec_policy(stream)->on(stream),
               predecessors,
               predecessors + graph.number_of_vertices,
               graph.number_of_vertices);

  VT level = 0;
  if (distances != nullptr) {
    thrust::fill(rmm::exec_policy(stream)->on(stream),
                 distances,
                 distances + graph.number_of_vertices,
                 std::numeric_limits<VT>::max());
  }

  // BFS communications wrapper
  BFSCommunicatorBCastReduce<VT, ET, WT> bfs_comm(handle, word_count);

  // 0. 'Insert' starting vertex in the input frontier
  input_frontier[start_vertex / BitsPWrd<unsigned>] = static_cast<unsigned>(1)
                                                      << (start_vertex % BitsPWrd<unsigned>);

  if (distances != nullptr) {
    // Set the distance of starting vertex to 0
    thrust::device_ptr<VT> dist(distances);
    dist[start_vertex] = VT{0};
  }

  do {
    // 1. Mark all input frontier vertices as visited
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      input_frontier.begin(),
                      input_frontier.end(),
                      visited.begin(),
                      visited.begin(),
                      bitwise_or());

    // 2. Clear out output frontier
    thrust::fill(rmm::exec_policy(stream)->on(stream),
                 output_frontier.begin(),
                 output_frontier.end(),
                 static_cast<unsigned>(0));

    // 3. Create output frontier from input frontier
    if (distances != nullptr) {
      // BFS Functor for frontier calculation
      detail::bfs_frontier_pred_dist<VT> bfs_op(
        output_frontier.data().get(), visited.data().get(), predecessors, distances, ++level);
      lb.run(bfs_op, input_frontier.data().get());
    } else {
      // BFS Functor for frontier calculation
      detail::bfs_frontier_pred<VT> bfs_op(
        output_frontier.data().get(), visited.data().get(), predecessors);
      lb.run(bfs_op, input_frontier.data().get());
    }

    // 3a. Combine output frontier from all GPUs
    bfs_comm.allreduce(output_frontier);

    // 4. 'Remove' all vertices in output frontier
    // that are already visited
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      visited.begin(),
                      visited.end(),
                      output_frontier.begin(),
                      output_frontier.begin(),
                      remove_visited());

    // 5. Use the output frontier as input for the next step
    input_frontier.swap(output_frontier);

    // 6. If all bits in input frontier are inactive then bfs is done
    frontier_not_empty[0] = 0;
    thrust::for_each(
      rmm::exec_policy(stream)->on(stream), input_frontier.begin(), input_frontier.end(), neq);
  } while (frontier_not_empty[0] == 1);

  // In place reduce to collect predecessors
  if (handle.comms_initialized()) {
    handle.get_comms().allreduce(predecessors,
                                 predecessors,
                                 graph.number_of_vertices,
                                 raft::comms::op_t::MIN,
                                 handle.get_stream());
  }

  // If the bfs loop does not assign a predecessor for a vertex
  // then its value will be graph.number_of_vertices. This needs to be
  // replaced by invalid vertex id to denote that a vertex does have
  // a predecessor
  thrust::replace(rmm::exec_policy(stream)->on(stream),
                  predecessors,
                  predecessors + graph.number_of_vertices,
                  graph.number_of_vertices,
                  cugraph::invalid_vertex_id<VT>::value);

  if (distances != nullptr) {
    // In place reduce to collect predecessors
    if (handle.comms_initialized()) {
      handle.get_comms().allreduce(distances,
                                   distances,
                                   graph.number_of_vertices,
                                   raft::comms::op_t::MIN,
                                   handle.get_stream());
    }
  }
}
#endif

}  // namespace opg

}  // namespace cugraph
