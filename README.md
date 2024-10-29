# Explain Then Predict

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) [![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/) [![TextAttack](https://img.shields.io/badge/TextAttack-20232A?logo=octopusdeploy&logoColor=D61F2A)](https://textattack.readthedocs.io/)

## Overview
This repository contains the source code for our BlackBoxNLP 2024 paper:

> _Enhancing adversarial robustness in Natural Language Inference using explanations_ (https://arxiv.org/abs/2409.07423)

In this work, we investigate whether the usage of intermediate explanations in the Natural Language Inference (NLI) task can serve as a model-agnostic defence strategy against adversarial attacks. Our claim is that the intermediate explanation can filter out potential noise that gets superimposed by the adversarial attack in the input pair (premise, hypothesis). Through extensive experimentation, we prove that conditioning the output label (entailment, contradiction, neutral) on an intermediate explanation that describes the inference relationship between the input premise and hypothesis, adversarial robustness is indeed achieved.