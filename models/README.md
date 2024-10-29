# Models

This directory contains the full code for training and evaluating all the models that are used in our experiments. Depending on whether the models involve natural language explanations or not, they are distinguished into _explanations-based_ (`expl-based`) and _explanations-free_ (`expl-free`), respectively.

The current directory includes the two following sub-directories:
 * `expl-free`: Refers to the training and evaluation of a BERT classifier NLI model which follow the traditional classification setup and does not leverage explanations and thus serves as baseline.
 * `expl-based`: Refers to the training and evaluation of the models that follow the proposed "Explain Then Predict" pipeline. According to this setup, the following apply:
    1. First, a _seq2seq_ model is fed with a pair (premise, hypothesis) and outputs an explanation that describes the inference relationship between premise and hypothesis.
    2. This explanation is fed to an _expl2label_ model which predicts the output label (entailment, contradiction, neutral) based on the input explanation.