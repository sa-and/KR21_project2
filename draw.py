from BayesNet import BayesNet
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    bifxml_path = "testing/lecture_example2.BIFXML"

    bn = BayesNet()
    bn.load_from_bifxml(bifxml_path)
    bn.draw_structure()

