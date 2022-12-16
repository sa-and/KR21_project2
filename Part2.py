from BNReasoner import BNReasoner
from BayesNet import BayesNet
import pandas as pd
import time
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main(folder_name, output_folder):
    node_counts = range(10, 51, 2)
    random_list = []
    md_list = []
    mf_list = []
    random_list2 = []
    md_list2 = []
    mf_list2 = []

    for node_count in node_counts:
        net = folder_name +"/network_"+str(node_count)+".XMLBIF"
        bn = BayesNet()
        bn.load_from_bifxml(net)
        bnr = BNReasoner(bn)
        print(node_count)
        methods = ['random', 'min_degree', 'min_fill']
        variable_list = bn.get_all_variables()

        for method in methods:
            start = time.time()
            outcome_mpe = bnr.MPE(pd.Series({variable_list[-1]: True, variable_list[-2]: True}),method)
            end = time.time()
            total_time = end - start
            if(method == 'random'):
                random_list.append(total_time)
            elif(method == 'min_degree'):
                md_list.append(total_time)
            elif(method == 'min_fill'):
                mf_list.append(total_time)

            start = time.time()
            outcome_map = bnr.MAP([variable_list[0], variable_list[1]], pd.Series({variable_list[-1]: True,variable_list[-2]: True}),method)
            end = time.time()
            total_time = end - start

            if(method == 'random'):
                random_list2.append(total_time)
            elif(method == 'min_degree'):
                md_list2.append(total_time)
            elif(method == 'min_fill'):
                mf_list2.append(total_time)


    df1 = pd.DataFrame({'Node_Count': node_counts, 'Random': random_list, 'MinDegree': md_list, 'MinFill': mf_list})
    df2 = pd.DataFrame({'Node_Count': node_counts, 'Random': random_list2, 'MinDegree': md_list2, 'MinFill': mf_list2})

    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    df1.to_excel(output_folder+'/MPE.xlsx', index=False)
    df2.to_excel(output_folder+'/MAP.xlsx', index=False)

if __name__ == "__main__":
    folder_name = "testing/Part_2"
    output_folder = "Output"
    main(folder_name, output_folder)