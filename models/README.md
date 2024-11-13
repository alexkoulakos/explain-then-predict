# Models

## Overview
This directory contains the full code for training and evaluating all the models that are used in our experiments. Depending on whether the models involve natural language explanations or not, they are distinguished into _explanations-based_ (`expl-based` directory) and _explanations-free_ (`expl-free` directory), respectively.

## Structure
The current directory includes the two following sub-directories:
 * `expl-free`: Refers to the training and evaluation of a BERT-based classifier for the NLI task. The model does not leverage explanations and thus serves as baseline.
 * `expl-based`: Refers to the training and evaluation of the models that leverage natural language explanations according to the *Explain Then Predict* pipeline.