import torch
import argparse

from text_processing import preprocessing
from DepGCN import Dependency_GCN

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

    # Simple One-hot encoding is applied. This can be replaced based on the choice of embedding language model.
    input_rep = tk2onehot(input_tk_list)
    
    dependency_list = list(set([x[1] for x in input_dep_list]))

    in_dim = len(input_tk_list)
    out_dim = len(input_rep[0])
    model = Dependency_GCN(in_dim=in_dim, out_dim=out_dim, dependency_list=dependency_list, 
                           num_layers = args.num_layers, reverse_case=args.reverse)

    output = model(input_rep, input_dep_list)

    print(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nlp_pipeline", default="stanza", type=str, help="NLP preprocessing pipeline.")
    parser.add_argument("--num_layers", default=1, type=int, help="The number of hidden layers of GCN.")
    parser.add_argument("--reverse", default=True, type=bool, help="Applying reverse dependency cases or not.")

    args = parser.parse_args()

    main(args)
