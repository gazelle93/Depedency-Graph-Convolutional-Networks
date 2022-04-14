# Overview
- Graph-Neural-Networks (GNN) represents a given graph where the nodes are linked with edges into a vector space. Relational-Graph-Convolutional-Networks (R-GCN) introduced relation dependant graph representation. The typed dependency of given text forms triples of words and dependency (t<sub>dep</sub>, dependency, t<sub>gov</sub>) is similar to the triples of nodes and edge in a graph (d<sub>i</sub>, rel, d<sub>j</sub>) where t is token and d is data. Therefore, this project aims to develop dependency-graph-convolutional-networks in order to represent the dependency relations of each word from given text. The output of the dependency-graph-convolutional-networks is the token-level representation of the sum of the token and its dependency.

# Brief description
- DepGCN.py
> Output format
> - output: The sum of token embedding itself and dependency relation that is connected to the governor. (list)
- text_processing.py
> Output format
> - input_tk_list (list): Tokens of given text based on the selected nlp pipeline.
> - input_dep_list (list): Dependency triple of given text based on the selected nlp pipeline.

# Prerequisites
- argparse
- torch
- stanza
- spacy

# Parameters
- nlp_pipeline(str, defaults to "stanza"): NLP preprocessing pipeline.
- reverse(bool, defaults to True): Applying reverse dependency cases (gov -> dep) or not.

# References
- R-GCN: Schlichtkrull, M., Kipf, T. N., Bloem, P., Berg, R. V. D., Titov, I., & Welling, M. (2018, June). Modeling relational data with graph convolutional networks. In European semantic web conference (pp. 593-607). Springer, Cham.
- Graph Convolutional Networks: Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907.
- Survey of Graph Neural Networks: Zhang, S., Tong, H., Xu, J., & Maciejewski, R. (2019). Graph convolutional networks: a comprehensive review. Computational Social Networks, 6(1), 1-23.
