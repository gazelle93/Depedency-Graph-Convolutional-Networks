import torch
import torch.nn as nn
import torch.nn.functional as F

class Dependency_GCN(nn.Module):
    def __init__(self, in_dim, out_dim, dependency_list, num_layers=1 ,reverse_case=True):
        super(Dependency_GCN, self).__init__()
        # dim: dimension of dependency weight
        # dependency_list: the entire dependency types
        # reverse_case (default=True): Considering not only the result of dependency representation but also the reversed dependency representation
        
        """
        - Text:
            My dog likes eating sausage.
            
        - Universal dependencies: 
            nmod:poss(dog-2, My-1)
            nsubj(likes-3, dog-2)
            root(ROOT-0, likes-3)
            xcomp(likes-3, eating-4)
            obj(eating-4, sausage-5)
            
        * Dependency can be presented as a directed graph
                  likes
                 /     \
           (nsubj)     (xcomp)
            |             |
            dog         eating
            |             |
           (nmod:poss)  (obj)
            |             |
            My          sausage
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        
        self.dependency_weight_list =[['self', nn.Linear(out_dim, out_dim)],['root', nn.Linear(out_dim, out_dim)]]
        self.reverse_case = reverse_case
        
        # Considering the result of dependency representation
        # (gov -> dep)
        if reverse_case == False:
            for label in dependency_list:
                self.dependency_weight_list.append([label, nn.Linear(out_dim, out_dim)])
            self.weights = nn.ModuleDict(self.dependency_weight_list)
            
            
        # Considering the result of dependency representation and the reversed dependency representation
        # (gov -> dep & dep -> gov)
        else:
            reversed_dep_list = []
            for dependency in dependency_list:
                reversed_dep_list.append(dependency+'_r')
            dependency_list = dependency_list + reversed_dep_list

            for label in dependency_list:
                self.dependency_weight_list.append([label, nn.Linear(out_dim, out_dim)])
            self.weights = nn.ModuleDict(self.dependency_weight_list)
            
    def message_passing(self, _input, dependency_triples):
        temp_tensor = torch.zeros(self.in_dim, self.out_dim)
    
        # token representation of itself
        for idx, tk_emb in enumerate(_input):
            temp_tensor[idx] = self.weights['self'](tk_emb)

        # (dependent, dependency, governor) representation
        if self.reverse_case == False:
            for dep_triple in dependency_triples:
                cur_governor = dep_triple[2]
                cur_dependency = dep_triple[1]
                cur_dependent = dep_triple[0]
                
                temp_tensor[cur_dependent] += self.weights[cur_dependency](_input[cur_governor].T)
        else:
            for dep_triple in dependency_triples:
                cur_governor = dep_triple[2]
                cur_dependency = dep_triple[1]
                cur_dependent = dep_triple[0]
                
                temp_tensor[cur_dependent] += self.weights[cur_dependency](_input[cur_governor].T)
                temp_tensor[cur_governor] += self.weights[cur_dependency+'_r'](_input[cur_dependent].T)
                
        return temp_tensor

    def forward(self, _input, dependency_triples):
        # _input: tokenized input text representation in vector space
        # dependency_triples: (dependent index, dependency, governor index)
        # * dependent and governor index follows the index of _input
        
        h_ = self.message_passing(_input, dependency_triples)
        
        if self.num_layers > 1:
            for _ in range(self.num_layers-2):
                h_ = self.message_passing(h_, dependency_triples)
            
        return h_
