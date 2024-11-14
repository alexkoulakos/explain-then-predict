# Adversarial attacks

## Overview
This directory contains the code for carrying out the adversarial attacks experiments against the fine-tuned explanations-based (`expl-based`) and explanations-free (`expl-free`) models. We choose to attack the victim models using the *TextFooler* and *BERT-attack* attack recipes, as they are state-of-the-art in NLP and, most importantly, achieve high-quality perturbations on samples drawn from the SNLI dataset.

## Structure
The current directory includes the two following sub-directories:
 * `expl-free`: Refers to the adversarial attacks that are carried out against a fine-tuned BERT-based classifier for the NLI task which does not leverage explanations and thus serves as baseline.
 * `expl-based`: Refers to the adversarial attacks that are carried out against fine-tuned models which leverage natural language explanations according to the *Explain Then Predict* pipeline.