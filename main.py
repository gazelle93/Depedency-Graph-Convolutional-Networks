from GraphLayers import Dependency_GCN
import torch
import stanza

def tk2onehot(_tk_list):
    tk_dim = len(_tk_list)
    tk2onehot = []
    for idx,_ in enumerate(_tk_list):
        temp = torch.zeros(tk_dim)
        temp[idx] = 1
        tk2onehot.append(temp)
    return tk2onehot

def main(args):
    nlp = stanza.Pipeline('en')
    sample_text = "My dog likes eating sausage"
    
    text = nlp(sample_text)
    
    input_tk_list = ["ROOT"]
    input_dep_list = []
    for sen in text.sentences:
        for tk in sen.tokens:
            tk_infor_dict = tk.to_dict()[0]
            cur_tk = tk_infor_dict["text"]

            cur_id = tk_infor_dict['id']
            cur_head = tk_infor_dict['head']
            cur_dep = tk_infor_dict["deprel"]

            cur_dep_triple = (cur_id, cur_dep, cur_head)
            input_tk_list.append(cur_tk)
            input_dep_list.append(cur_dep_triple)

    """
    print(input_tk_list)
    -> ['ROOT', 'My', 'dog', 'likes', 'eating', 'sausage']

    print(input_dep_list)
    -> [(1, 'nmod:poss', 2), (2, 'nsubj', 3), (3, 'root', 0), (4, 'xcomp', 3), (5, 'obj', 4)]
    """

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

    parser.add_argument("--reverse", default=True, type=boolean, help="Applying reverse dependency cases or not.")

    args = parser.parse_args()

    main(args)
