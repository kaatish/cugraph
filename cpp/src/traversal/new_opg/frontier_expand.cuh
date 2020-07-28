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

namespace mg {

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
void frontier_expand(
    raft::handle_t const &handle,
    cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
    rmm::device_vector<vertex_t> &input_frontier,
    rmm::device_vector<unsigned> &visited_bmap,
    rmm::device_vector<unsigned> &isolated_bmap,
    vertex_t level,
    vertex_t *distances,
    vertex_t *predecessors,
    rmm::device_vector<vertex_t> &output_frontier) {
}

template <typename vertex_t, typename edge_t, typename weight_t>
class FrontierExpand {
  raft::handle_t const &handle_;
  cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph_;
  vertex_t vertex_begin_;
  vertex_t vertex_end_;
  rmm::device_vector<edge_t> frontier_vertex_offset_;
  rmm::device_vector<edge_t> frontier_vertex_bucket_offset_;

  FrontierExpand(raft::handle_t const &handle,
      cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph)
    : handle_(handle), graph_(graph)
  {
    bool is_opg = (handle.comms_initialized() && (graph.local_vertices != nullptr) &&
                   (graph.local_offsets != nullptr));
    if (is_opg) {
      vertex_begin_ = graph.local_offsets[handle_.get_comms().get_rank()];
      vertex_end_   = graph.local_offsets[handle_.get_comms().get_rank()] +
                    graph.local_vertices[handle_.get_comms().get_rank()];
    } else {
      vertex_begin_ = 0;
      vertex_end_   = graph.number_of_vertices;
    }
    frontier_vertex_offset_.resize(vertex_end_ - vertex_begin_ + 1);
    frontier_vertex_bucket_offset_.resize(
        (number_of_edges / TOP_DOWN_EXPAND_DIMX + 1) * NBUCKedge_tS_PER_BLOCK + 2));
  }

};

} //namespace detail

} // namespace mg

} // namespace cugraph
