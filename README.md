# Overview
- This project aims to develop dependency graph convolutional networks in order to represent the dependency relations of each word from given text. 

# Brief description
- Dep_GCN.py
> Output format
> - output: The sum of token embedding itself and dependency relation that is connected to the governor. (list)
- text_processing.py
> Output format
> - input_tk_list: Tokens of given text based on the selected nlp pipeline. (list)
> - input_dep_list: Dependency triple of given text based on the selected nlp pipeline. (list)

# Prerequisites
> - argparse
> - torch
> - stanza
> - spacy

# Parameters
> - nlp_pipeline(str, defaults to "stanza"): NLP preprocessing pipeline.
> - reverse(bool, defaults to True): Applying reverse dependency cases (gov -> dep) or not.

# References
- Graph Neural Networks: Scarselli, Franco, et al. "The graph neural network model." IEEE transactions on neural networks 20.1 (2008): 61-80.
