import torch
import argparse

from text_processing import preprocessing
from GraphLayers import Dependency_GCN

def tk2onehot(_tk_list):
    tk_dim = len(_tk_list)
    tk2onehot = []
    for idx,_ in enumerate(_tk_list):
        temp = torch.zeros(tk_dim)
        temp[idx] = 1
        tk2onehot.append(temp)
    return tk2onehot

def main(args):
    sample_text = "My dog likes eating sausage"
    input_tk_list, input_dep_list = preprocessing(sample_text, args.nlp_pipeline)

    input_rep = tk2onehot(input_tk_list)
    dependency_list = [x[1] for x in input_dep_list]

    model = Dependency_GCN(dim=len(input_tk_list), dependency_list=dependency_list, reverse_case=args.reverse)
    """
    print(model)
    -> Dependency_GCN(
      (weights): ModuleDict(
        (self): Linear(in_features=6, out_features=6, bias=True)
        (nmod:poss): Linear(in_features=6, out_features=6, bias=True)
        (nsubj): Linear(in_features=6, out_features=6, bias=True)
        (root): Linear(in_features=6, out_features=6, bias=True)
        (xcomp): Linear(in_features=6, out_features=6, bias=True)
        (obj): Linear(in_features=6, out_features=6, bias=True)
        (nmod:poss_r): Linear(in_features=6, out_features=6, bias=True)
        (nsubj_r): Linear(in_features=6, out_features=6, bias=True)
        (root_r): Linear(in_features=6, out_features=6, bias=True)
        (xcomp_r): Linear(in_features=6, out_features=6, bias=True)
        (obj_r): Linear(in_features=6, out_features=6, bias=True)
      )
    )
    """
    output = model(input_rep, input_dep_list)

    print(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nlp_pipeline", default="stanza", type=str, help="NLP preprocessing pipeline.")
    parser.add_argument("--reverse", default=True, type=bool, help="Applying reverse dependency cases or not.")

    args = parser.parse_args()

    main(args)
