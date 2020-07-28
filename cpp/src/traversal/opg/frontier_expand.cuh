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

#include <raft/cudart_utils.h>
#include <graph.hpp>
#include "bfs_kernels.cuh"
#include "../traversal_common.cuh"

namespace cugraph {

namespace opg {

namespace detail {


template <typename vertex_t, typename edge_t>
__global__ void
write_vertex_degree(
    edge_t * offsets,
    vertex_t * input_frontier,
    vertex_t vertex_begin,
    edge_t total_vertex_count,
    edge_t * frontier_vertex_degree) {
  edge_t id = threadIdx.x + (blockIdx.x * blockDim.x);
  if (id < total_vertex_count) {
    vertex_t source_id = input_frontier[id];
    vertex_t loc = source_id - vertex_begin;
    frontier_vertex_degree[id] = offsets[loc + 1] - offsets[loc];
  }
}

template <typename vertex_t, typename edge_t>
void get_frontier_vertex_offsets(
    edge_t * offsets,
    rmm::device_vector<vertex_t> &input_frontier,
    vertex_t vertex_begin,
    edge_t vertex_begin_index,
    edge_t vertex_end_index,
    rmm::device_vector<edge_t> &frontier_vertex_offset,
    cudaStream_t stream) {
    //frontier_vertex_offset_.resize(vertex_end_index - vertex_begin_index + 1);
    frontier_vertex_offset[0] = 0;
  frontier_vertex_offset.resize(vertex_end_index - vertex_begin_index + 1);
  const int BLOCK_SIZE = 128;
  int block_count = raft::div_rounding_up_unsafe(
      vertex_end_index - vertex_begin_index, BLOCK_SIZE);
  write_vertex_degree<<<block_count, BLOCK_SIZE, 0, stream>>>(
      offsets,
      input_frontier.data().get() + vertex_begin_index,
      vertex_begin,
      vertex_end_index - vertex_begin_index,
      frontier_vertex_offset.data().get() + 1);
  thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
      frontier_vertex_offset.begin() + 1, frontier_vertex_offset.end(),
      frontier_vertex_offset.begin() + 1);
      CHECK_CUDA(stream);
  std::cout<<"frontier_vertex_offset :\n";
  thrust::copy(frontier_vertex_offset.begin(), frontier_vertex_offset.end(),
      std::ostream_iterator<edge_t>(std::cout, " "));
      CHECK_CUDA(stream);
  std::cout<<"Done\n";
}

template <typename edge_t>
__global__ void
compute_bucket_offsets_kernel(const edge_t* frontier_degrees_exclusive_sum,
                              edge_t* bucket_offsets,
                              const edge_t frontier_size,
                              edge_t total_degree)
{
  edge_t end =
    ((total_degree - 1 + TOP_DOWN_EXPAND_DIMX) /
     TOP_DOWN_EXPAND_DIMX * NBUCKETS_PER_BLOCK + 1);

  for (edge_t bid = blockIdx.x * blockDim.x + threadIdx.x; bid <= end;
       bid += gridDim.x * blockDim.x) {
    edge_t eid = min(bid * TOP_DOWN_BUCKET_SIZE, total_degree - 1);

    bucket_offsets[bid] =
      cugraph::detail::traversal::binsearch_maxle<edge_t>(
          frontier_degrees_exclusive_sum, eid, (edge_t)0, frontier_size - 1);
  }
}

template <typename edge_t>
void compute_bucket_offsets(rmm::device_vector<edge_t> &frontier_offsets,
                            rmm::device_vector<edge_t> &frontier_bucket_offsets,
                            edge_t total_frontier_edge_count,
                            cudaStream_t stream)
{
  dim3 grid, block;
  block.x = COMPUTE_BUCKET_OFFSETS_DIMX;

  grid.x =
    min(static_cast<edge_t>(MAXBLOCKS),
        ((total_frontier_edge_count - 1 + TOP_DOWN_EXPAND_DIMX) /
         TOP_DOWN_EXPAND_DIMX * NBUCKETS_PER_BLOCK + 1 + block.x - 1) /
        block.x);
  //Total number of vertices in frontier = frontier_offsets.size() - 1
  compute_bucket_offsets_kernel<<<grid, block, 0, stream>>>(
    frontier_offsets.data().get(),
    frontier_bucket_offsets.data().get(),
    static_cast<edge_t>(frontier_offsets.size() - 1),
    total_frontier_edge_count);
  CHECK_CUDA(stream);
}

template <typename T>
struct TupleMinus : thrust::unary_function<thrust::tuple<T, T> const&, T> {
__device__
T operator()(const thrust::tuple<T, T>& in) {
  return thrust::get<0>(in) - thrust::get<1>(in);
}
};

template <typename vertex_t>
struct FindBoundary {
  vertex_t begin_;
  vertex_t end_;
  vertex_t * frontier_;
  vertex_t * boundary_;

  FindBoundary(
      vertex_t begin,
      vertex_t end,
      vertex_t * frontier,
      vertex_t * boundary) :
    begin_(begin),
    end_(end),
    frontier_(frontier),
    boundary_(boundary) {}

  template <typename T>
  __device__
  void operator()(const T& id) {
    if ((frontier_[id-1] < begin_) &&
      (frontier_[id] >= begin_)) {
      boundary_[0] = id;
    }
    if ((frontier_[id-1] < end_) &&
      (frontier_[id] >= end_)) {
      //We want the first index which is >= end_
      boundary_[1] = id;
    }
  }
};

template <typename vertex_t, typename edge_t, typename weight_t>
class FrontierOperator {
  raft::handle_t const &handle_;
  cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph_;
  rmm::device_vector<unsigned> const &isolated_bmap_;
  vertex_t vertex_begin_;
  vertex_t vertex_end_;
  edge_t edge_count_;
  rmm::device_vector<edge_t> frontier_vertex_offset_;
  rmm::device_vector<edge_t> frontier_vertex_bucket_offset_;
  rmm::device_vector<edge_t> boundary_;
  rmm::device_vector<edge_t> output_frontier_size_;
  bool is_opg_;

public:
  FrontierOperator(
      raft::handle_t const &handle,
      cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
      rmm::device_vector<unsigned> const &isolated_bmap)
    : handle_(handle), graph_(graph), isolated_bmap_(isolated_bmap)
  {
    is_opg_ = handle.comms_initialized() &&
      (graph_.local_vertices != nullptr) &&
      (graph_.local_offsets != nullptr);
    if (is_opg_) {
      vertex_begin_ = graph_.local_offsets[handle_.get_comms().get_rank()];
      vertex_end_   = graph_.local_offsets[handle_.get_comms().get_rank()] +
                    graph_.local_vertices[handle_.get_comms().get_rank()];
      edge_count_ = graph_.local_edges[handle_.get_comms().get_rank()];
    } else {
      vertex_begin_ = 0;
      vertex_end_   = graph_.number_of_vertices;
      edge_count_ = graph_.number_of_edges;
    }
    frontier_vertex_offset_.resize(vertex_end_ - vertex_begin_ + 1);
    frontier_vertex_bucket_offset_.resize(
        (edge_count_ / TOP_DOWN_EXPAND_DIMX + 1) * NBUCKETS_PER_BLOCK + 2);
    boundary_.resize(2);
    output_frontier_size_.resize(1);
  }

  void run(
      rmm::device_vector<vertex_t>& input_frontier,
      vertex_t level,
      rmm::device_vector<vertex_t>& output_frontier,
      rmm::device_vector<unsigned> &visited_bmap,
      vertex_t *distances,
      vertex_t *predecessors)
  {
    if (input_frontier.size() == 0) { return; }
    edge_t vertex_begin_index = edge_t{0};
    edge_t vertex_end_index = static_cast<edge_t>(input_frontier.size());

    cudaStream_t stream = handle_.get_stream();

    if (input_frontier.size() == 1) {
      auto source = input_frontier[0];
      if ((source >= vertex_begin_) && (source < vertex_end_)) {
        vertex_begin_index = 0;
        vertex_end_index = 1;
      } else {
        return;
      }
    }
    else if (is_opg_) {
      //If working in opg context, figure out which section
      //of the frontier you want to launch the operator on
      //Since the frontier might contain vertices that are not
      //local to the GPU only a few vertices will need to run
      //This function also requires input_frontier to be sorted
      std::vector<edge_t> h_boundary(2,
          static_cast<edge_t>(input_frontier.size()));
      boundary_ = h_boundary;
      thrust::for_each(rmm::exec_policy(stream)->on(stream),
          thrust::make_counting_iterator<edge_t>(1),
          thrust::make_counting_iterator<edge_t>(input_frontier.size()),
          FindBoundary<vertex_t>(
            vertex_begin_,
            vertex_end_,
            input_frontier.data().get(),
            boundary_.data().get()));
      //boundary[0] is the index of input_frontier where
      //first vertex >= vertex_begin_ exists
      //boundary[1] is the index of input_frontier where
      //first vertex >= vertex_end_ exists
      vertex_begin_index = boundary_[0];
      vertex_end_index = boundary_[1];
    }

    //Return if no valid vertices are in the range then exit
    if (vertex_begin_index == vertex_end_index) { return; }

    //auto offset_ptr = thrust::device_pointer_cast(graph_.offsets);
    thrust::device_ptr<edge_t> offset_ptr(graph_.offsets);
    auto degree_iter =
      thrust::make_transform_iterator(thrust::make_zip_iterator(
            thrust::make_tuple(offset_ptr+1, offset_ptr)), TupleMinus<edge_t>());
    auto transform_iter =
        thrust::make_transform_iterator(input_frontier.begin(),
          [=] __device__ (vertex_t id) { return id - vertex_begin_; });
    auto perm_iter = thrust::make_permutation_iterator(degree_iter,
        transform_iter);

    //if (true) {
    //  std::cout<<"OUTPUT input_frontier\n";
    //  std::cout<<offset_ptr[0]<<" "<<offset_ptr[1]<<"\n";
    //  for (int i = vertex_begin_index; i < vertex_end_index; ++i) {
    //    std::cout<<offset_ptr[i]<<"\t";
    //    std::cout<<input_frontier[i]<<"\t"<<transform_iter[i]<<"\n";
    //  }
    //  CHECK_CUDA(stream);
    //  std::cout<<"Done\n";
    //}

    //Get prefix sum, and launch bs lb
    //Get the degree of all vertices in the index range
    //[vertex_begin_index, vertex_end_index)
    {
      get_frontier_vertex_offsets(
          graph_.offsets,
          input_frontier,
          vertex_begin_,
          vertex_begin_index, vertex_end_index,
          frontier_vertex_offset_,
          stream);
    //thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
    //    perm_iter + vertex_begin_index, perm_iter + vertex_end_index,
    //    frontier_vertex_offset_.begin() + 1);
    }
      CHECK_CUDA(stream);

    //Total number of edges to be worked on
    edge_t total_frontier_edge_count =
      frontier_vertex_offset_[frontier_vertex_offset_.size() - 1];

    //If total number of edges to be worked on is 0 then exit
    if (total_frontier_edge_count == 0) { return; }

    compute_bucket_offsets(
        frontier_vertex_offset_,
        frontier_vertex_bucket_offset_,
        total_frontier_edge_count,
        stream);
    CHECK_CUDA(stream);
    output_frontier.resize(graph_.number_of_vertices);
    output_frontier_size_[0] = 0;
    frontier_expand<vertex_t, edge_t, weight_t>(
        graph_,
        input_frontier,
        vertex_begin_,
        vertex_begin_index,
        vertex_end_index,
        isolated_bmap_,
        level,
        total_frontier_edge_count,
        frontier_vertex_offset_,
        frontier_vertex_bucket_offset_,
        visited_bmap,
        output_frontier,
        output_frontier_size_,
        distances,
        predecessors,
        stream);
    output_frontier.resize(output_frontier_size_[0]);
    std::cout<<"output frontier size "<<output_frontier_size_[0]<<"\n";

  }
};

} //namespace detail

} // namespace opg

} // namespace cugraph
