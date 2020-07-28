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

namespace cugraph {

namespace opg {

namespace detail {

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
      cugraph::GraphCSRView<VT, ET, WT> const &graph)
    : handle_(handle), graph_(graph)
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
        (edge_count_ / TOP_DOWN_EXPAND_DIMX + 1) * NBUCKETS_PER_BLOCK + 2));
    boundary_.resize(2);
    output_frontier_size_.resize(1);
  }

  template <typename Operator>
  void run(Operator op,
      rmm::device_vector<vertex_t>& input_frontier,
      rmm::device_vector<vertex_t>& output_frontier)
  {
    if (input_frontier.size() == 0) { return; }
    edge_t vertex_begin_index = edge_t{0};
    edge_t vertex_end_index = edge_t{input_frontier.size()};

    cudaStream_t stream = handle_.get_stream();

    //If working in opg context, figure out which section
    //of the frontier you want to launch the operator on
    //Since the frontier might contain vertices that are not
    //local to the GPU only a few vertices will need to run
    //This function also requires input_frontier to be sorted
    if (is_opg_) {
      std::vector<edge_t> h_boundary=
      {input_frontier.size(), input_frontier.size()};
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

    auto offset_ptr = thrust::device_pointer_cast(graph_.offsets);
    auto degree_iter =
      thrust::make_transform_iterator(thrust::make_zip_iterator(
            thrust::make_tuple(offset_ptr+1, offset_ptr)), TupleMinus());
    auto perm_iter = thrust::make_permutation_iterator(degree_iter,
        thrust::make_transform_iterator(input_frontier.begin(),
          [=] __device__ (vertex_t id) { return id - vertex_begin_; }));

    //Get prefix sum, and launch bs lb
    //Get the degree of all vertices in the index range
    //[vertex_begin_index, vertex_end_index)
    frontier_vertex_offset_.resize(vertex_end_index - vertex_begin_index + 1);
    frontier_vertex_offset_[0] = 0;
    thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
        perm_iter + vertex_begin_index, perm_iter + vertex_end_index,
        frontier_vertex_offset_.begin() + 1);

    //Total number of edges to be worked on
    edge_t total_frontier_edge_count =
      frontier_vertex_offset_[frontier_vertex_offset_.size() - 1];

    //If total number of edges to be worked on is 0 then exit
    if (total_frontier_edge_count == 0) { return; }

    compute_bucket_offset(
        frontier_vertex_offset_,
        frontier_vertex_bucket_offset_,
        total_frontier_edge_count,
        stream);
    output_frontier.resize(graph_.number_of_vertices);
    frontier_expand(
        graph_,
        input_frontier,
        vertex_begin_index,
        vertex_end_index,
        frontier_vertex_offset_,
        frontier_vertex_bucket_offset_,
        visited_bmap,
        isolated_bmap,
        output_frontier,
        output_frontier_size_,
        distances,
        predecessors,
        stream);
    output_frontier.resize(output_frontier_size_[0]);

  }
};

} //namespace detail

} // namespace opg

} // namespace cugraph
