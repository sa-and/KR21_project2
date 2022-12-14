from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFWriter

def get_and_save():
    for nnodes in range(5, 100, 5):
        rand = BayesianNetwork.get_random(n_nodes=nnodes, n_states=2)
        writer = XMLBIFWriter(rand)
        fname = f'rand_bn_nodes_{nnodes}.BIFXML'
        writer.write_xmlbif(fname)


if __name__ == '__main__':
    get_and_save()