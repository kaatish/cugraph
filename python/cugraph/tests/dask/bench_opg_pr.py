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
    #input_data_path = r"../../../../datasets/GAP/GAP-twitter.csv"
    input_data_path = r"/home/aatish/workspace/cugraph/datasets/netscience.csv"
    chunksize = dcg.get_chunksize(input_data_path)
    ddf = dask_cudf.read_csv(input_data_path, chunksize=chunksize,
                             delimiter=' ',
                             names=['src', 'dst'],
                             dtype=['int32', 'int32'])
    #ddf = ddf.sort_values(by ='src')
    #print(ddf.compute())
    dg = cugraph.DiGraph()
    dg.from_dask_cudf_edgelist(ddf)
    t3 = time.time()
    #Pre compute local data
    dg.compute_local_data(by='dst')
    t4 = time.time()
    result_pr = dcg.pagerank(dg)
    t5 = time.time()
    print("---------- Comms Intialization Time: ", t2-t1, "s ----------")
    print("---------- read_csv + compute_local_data Time: ", t4-t3, "s ----------")
    print("---------- Pagerank Call Time: ", t5-t4, "s ----------")
    print(result_pr)
    # Sort, descending order
    pr_sorted_df = result_pr.sort_values('pagerank',ascending=False)
    # Print the Top 3
    print(pr_sorted_df.head(3))
    Comms.destroy()
    client.close()
    cluster.close()
