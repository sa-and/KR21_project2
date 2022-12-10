from tqdm import tqdm
from pathlib import Path
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFWriter

# -------------------------------- SETTINGS --------------------------------- #
FOLDER_PATH = 'testing/test_set1/'
NUM_NETWORKS = 10      # The number of networks for each size
SIZE_RANGE = [3, 10]    # The range for the number of nodes in the network (inclusive)
EDGE_PROB = 0.5         # The probability of an edge between any two nodes in the topologically sorted DAG.
# --------------------------------------------------------------------------- #

# Generate the test set
for size in tqdm(range(SIZE_RANGE[0], SIZE_RANGE[1] + 1)):
    dest_dir = f"{FOLDER_PATH}{size}/"
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    for num in range(NUM_NETWORKS):
        # Generate a random bayesian network
        model = BayesianNetwork.get_random(n_nodes=size, edge_prob=EDGE_PROB, n_states=2)

        # The 'get_random' function produces a model with integer node labels, 
        # which produces an error when trying to save it in XMLBIF format. As 
        # a workaround, create a new BN using the same edges with string labels,
        # and randomly initialise all CPDs
        new_model = BayesianNetwork()

        for node in model.nodes():
            new_model.add_node(str(node))

        for node1, node2 in model.edges():
            new_model.add_edge(str(node1), str(node2))

        new_model.get_random_cpds(n_states=2, inplace=True)

        # Save the new BN in XMLBIF format at the specified location
        writer = XMLBIFWriter(new_model)
        writer.write_xmlbif(f"{dest_dir}{num}.BIFXML")

        




