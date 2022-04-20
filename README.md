# Overview
- Graph-Neural-Networks (GNN) represents a given graph where the nodes are linked with edges into a vector space. Relational-Graph-Convolutional-Networks (R-GCN) introduced relation dependant graph representation. The typed dependency of given text forms triples of words and dependency (t<sub>dep</sub>, dependency, t<sub>gov</sub>) is similar to the triples of nodes and edge in a graph (d<sub>i</sub>, rel, d<sub>j</sub>) where t is token and d is data. Therefore, this project aims to implement pytorch dependency-graph-convolutional-networks in order to represent the dependency relations of each word from given text. The output of the dependency-graph-convolutional-networks is the token-level representation of the sum of the token and its dependency.

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
- num_layers(int, defaults to 1): The number of hidden layers of the architecture.
- reverse(bool, defaults to True): Applying reverse dependency cases (gov -> dep) or not.

# References
- Graph Convolutional Networks (GCN): Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907.
- Relational-Graph-Convolutional-Networks (R-GCN): Schlichtkrull, M., Kipf, T. N., Bloem, P., Berg, R. V. D., Titov, I., & Welling, M. (2018, June). Modeling relational data with graph convolutional networks. In European semantic web conference (pp. 593-607). Springer, Cham.
- Stanza: Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). Stanza: A Python natural language processing toolkit for many human languages. arXiv preprint arXiv:2003.07082.
- Spacy: Matthew Honnibal and Ines Montani. 2017. spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. To appear (2017).
