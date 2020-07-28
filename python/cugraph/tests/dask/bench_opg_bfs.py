import cugraph.dask as dcg
import cugraph.comms as Comms
from dask.distributed import Client
import gc
import cugraph
import dask_cudf
import cudf
from dask_cuda import LocalCUDACluster
def get_n_gpus():
    import os
    try:
        return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    except KeyError:
        return len(os.popen("nvidia-smi -L").read().strip().split("\n"))

def test_dask_pagerank():
    print("****************** GPUS: ", get_n_gpus(), " *******************")
    #rmm.reinitialize(managed_memory=True)
    gc.collect()
    cluster = LocalCUDACluster() #rmm_managed_memory=True)
    client = Client(cluster)
    import time
    t1 = time.time()
    Comms.initialize()
    t2 = time.time()
    #input_data_path = r"/home/aatish/workspace/datasets/GAP-road.csv"
    input_data_path = r"/home/aatish/workspace/cugraph/datasets/karate.csv"
    chunksize = dcg.get_chunksize(input_data_path)
    ddf = dask_cudf.read_csv(input_data_path, chunksize=chunksize,
                             delimiter=' ',
                             names=['src', 'dst'],
                             dtype=['int32', 'int32'])

    df = cudf.read_csv(input_data_path,
                       delimiter=' ',
                       names=['src', 'dst', 'value'],
                       dtype=['int32', 'int32', 'float32'])

    g = cugraph.DiGraph()
    g.from_cudf_edgelist(df, 'src', 'dst')

    #ddf = ddf.sort_values(by ='src')
    #print(ddf.compute())
    dg = cugraph.DiGraph()
    dg.from_dask_cudf_edgelist(ddf)
    t3 = time.time()
    #Pre compute local data
    dg.compute_local_data(by='src')
    t4 = time.time()
    result_dist = dcg.bfs(dg, 33, True)
    t5 = time.time()
    expected_dist = cugraph.bfs(g, 33)
    t6 = time.time()
    print("---------- Comms Intialization Time: ", t2-t1, "s ----------")
    print("---------- read_csv + compute_local_data Time: ", t4-t3, "s ----------")
    print("---------- MG BFS Call Time: ", t5-t4, "s ----------")
    print("---------- SG BFS Call Time: ", t6-t5, "s ----------")
    print(result_dist)
    # Sort, descending order
    #pr_sorted_df = result_pr.sort_values('pagerank',ascending=False)
    # Print the Top 3
    #print(pr_sorted_df.head(3))
    Comms.destroy()
    client.close()
    cluster.close()
