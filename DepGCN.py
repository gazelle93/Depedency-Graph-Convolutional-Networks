import torch
import torch.nn as nn
import torch.nn.functional as F

class Dependency_GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dependency_list, reverse_case=True, dropout_rate=0.1):
        super(Dependency_GCNLayer, self).__init__()
        # dim: dimension of dependency weight
        # dependency_list: the entire dependency types
        # reverse_case (default=True): Considering not only the result of dependency representation but also the reversed dependency representation
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        
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
    
    def forward(self, _input, dependency_triples, is_dropout=True):
        if is_dropout:
            return self.dropout(self.relu(self.message_passing(_input, dependency_triples)))
        return self.relu(self.message_passing(_input, dependency_triples))
    
class Dependency_GCN(nn.Module):
    def __init__(self, in_dim, out_dim, dependency_list, num_layers=1 ,reverse_case=True):
        super(Dependency_GCN, self).__init__()
        self.num_layers = num_layers
        self.gcn_layer = []
        for i in range(num_layers):
            self.gcn_layer.append(Dependency_GCNLayer(in_dim, out_dim, dependency_list,reverse_case))

    def forward(self, _input, dependency_triples):
        h_ = self.gcn_layer[0](_input, dependency_triples, False)
        if self.num_layers > 1:
            for i in range(self.num_layers-1):
                h_ = self.gcn_layer[i+1](h_, dependency_triples)
                
        return h_
