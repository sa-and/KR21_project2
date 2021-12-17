from BNReasoner import BNReasoner
from BayesNet import BayesNet

filePath = 'testing/dog_problem.BIFXML' ## filepath

bnet = BayesNet() ## make empty network
bnet.load_from_bifxml(file_path=filePath) ## fill that bitch up with data

bnet.get_interaction_graph() ## get interaction graph (idk what this does tbh)
bnet.draw_structure() ## draw the graph




