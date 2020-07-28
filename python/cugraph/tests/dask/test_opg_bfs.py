# Copyright (c) 2020, NVIDIA CORPORATION.
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

import cugraph.dask as dcg
import cugraph.comms as Comms
from dask.distributed import Client
import gc
import cugraph
import dask_cudf
import cudf
from dask_cuda import LocalCUDACluster


def test_dask_pagerank():
    gc.collect()
    cluster = LocalCUDACluster()
    client = Client(cluster)
    Comms.initialize()

    #input_data_path = r"../datasets/karate.csv"
    #input_data_path = r"/home/aatish/workspace/cugraph/datasets/netscience.csv"
    #input_data_path = r"/home/aatish/workspace/cugraph/datasets/netscience.csv"
    input_data_path = r"/home/aatish/workspace/datasets/GAP-road.csv"
    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(input_data_path, chunksize=chunksize,
                             delimiter=' ',
                             names=['src', 'dst', 'value'],
                             dtype=['int32', 'int32', 'float32'])

    df = cudf.read_csv(input_data_path,
                       delimiter=' ',
                       names=['src', 'dst', 'value'],
                       dtype=['int32', 'int32', 'float32'])

    g = cugraph.DiGraph()
    g.from_cudf_edgelist(df, 'src', 'dst')

    dg = cugraph.DiGraph()
    dg.from_dask_cudf_edgelist(ddf)

    # Pre compute local data
    # dg.compute_local_data(by='dst')

    expected_dist = cugraph.bfs(g, 0)
    result_dist = dcg.bfs(dg, 0, True)

    err = 0

    assert len(expected_dist) == len(result_dist)
    for i in range(len(result_dist)):
        if(result_dist['distance'].iloc[i] != expected_dist['distance'].iloc[i]):
            err = err + 1
            res = str(result_dist['distance'].iloc[i])
            gold = str(expected_dist['distance'].iloc[i])
            res_pred = str(result_dist['predecessor'].iloc[i])
            print(str(i) + ' ' + res + ' ' + gold + ' ' + res_pred)
    assert err == 0

    Comms.destroy()
    client.close()
    cluster.close()
