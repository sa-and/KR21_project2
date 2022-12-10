import time
from datetime import timedelta
import contextlib
import joblib
from tqdm import tqdm
from pathlib import Path
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFWriter

# -------------------------------- SETTINGS --------------------------------- #
FOLDER_PATH = 'testing/test_set1/'
NUM_NETWORKS = 10      # The number of networks for each size
SIZE_RANGE = [3, 30]   # The range for the number of nodes in the network (inclusive)
EDGE_PROB = 0.5        # The probability of an edge between any two nodes in the topologically sorted DAG.
# --------------------------------------------------------------------------- #

# Function to be run in parallel
def gen_bns(size):
    dest_dir = f"{FOLDER_PATH}{size}/"
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    for num in range(NUM_NETWORKS):
        # Generate a random bayesian network
        model = BayesianNetwork.get_random(n_nodes=size, edge_prob=EDGE_PROB, n_states=2)

        # The 'get_random' function produces a model with integer node labels, 
        # which produces an error when trying to save it in XMLBIF format. As 
        # a workaround, create a new BN using the same nodes and edges with 
        # string labels, and randomly initialise all CPDs
        new_model = BayesianNetwork()

        for node in model.nodes():
            new_model.add_node(str(node))

        for node1, node2 in model.edges():
            new_model.add_edge(str(node1), str(node2))

        new_model.get_random_cpds(n_states=2, inplace=True)

        # Save the new BN in XMLBIF format at the specified location
        writer = XMLBIFWriter(new_model)
        writer.write_xmlbif(f"{dest_dir}{num}.BIFXML")

# Context manager to integrate tqdm progress bar
# credit: https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# Generate the test set
size_range = range(SIZE_RANGE[0], SIZE_RANGE[1] + 1)

tic = time.perf_counter()
with tqdm_joblib(tqdm(desc="Generate test set", total=len(size_range))) as progress_bar:
    joblib.Parallel(n_jobs=2)(joblib.delayed(gen_bns)(size) for size in size_range)
toc = time.perf_counter()

seconds_elapsed = toc - tic

print(f"Elapsed time: {timedelta(seconds=seconds_elapsed)}")


        




