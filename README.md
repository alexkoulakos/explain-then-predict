# Adversarial robustness under the "Explain Then Predict" setting

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) [![Transformers](https://img.shields.io/badge/Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/) [![TextAttack](https://img.shields.io/badge/TextAttack-20232A?logo=octopusdeploy&logoColor=D61F2A)](https://textattack.readthedocs.io/)

## Overview
This repository contains the source code for our BlackBoxNLP 2024 @ EMNLP paper:

[_Enhancing adversarial robustness in Natural Language Inference using explanations_](https://aclanthology.org/2024.blackboxnlp-1.7)

In this work, we investigate whether the usage of intermediate explanations in the Natural Language Inference (NLI) task can serve as a model-agnostic defence strategy against adversarial attacks. Our claim is that the intermediate explanation can filter out potential noise superimposed by the adversarial attack in the input pair (premise, hypothesis). Through extensive experimentation, we prove that conditioning the output label (entailment, contradiction, neutral) on an intermediate explanation that describes the inference relationship between the input premise and hypothesis, adversarial robustness is indeed achieved.

## Project structure
The repo is organized in the following core directories:
 * `fine-tuning`: Includes the code for training and evaluating all the models that are used in our experiments. See the README file located in the `fine-tuning` directory for more details.
 * `adversarial_attacks`: Includes the code for performing adversarial attacks against the aforementioned models. See the README file located in the `adversarial_attacks` directory for more details.

## Installation
1. Create a local copy of the repo: `git clone https://github.com/alexkoulakos/explain-then-predict.git`
2. Navigate to the root directory: `cd explain-then-predict`
3. Create a virtual environment called _venv_: `virtualenv --system-site-packages venv`
4. Activate the virtual environment: `src venv/bin/activate` (for Linux/MacOS) or `./venv/Scripts/activate.ps1` (for Windows)
5. Install necessary dependancies: `pip install -r requirements.txt`

## Support and Issues
If you encounter any issues, bugs, or have questions, please feel free to open an issue on GitHub. Describe the problem you encountered, including:

* A clear description of the issue or bug
* Steps to reproduce the issue (if applicable)
* Any relevant error messages or screenshots
* Details about your environment (Python version, OS, library versions)

We’ll do our best to respond quickly and help resolve any problems.

## Citation
If you use our findings in your work, don't forget to cite our paper:

```
@inproceedings{koulakos-etal-2024-enhancing,
    title = "Enhancing adversarial robustness in Natural Language Inference using explanations",
    author = "Koulakos, Alexandros and Lymperaiou, Maria and Filandrianos, Giorgos and Stamou, Giorgos",
    editor = "Belinkov, Yonatan and Kim, Najoung and Jumelet, Jaap and Mohebbi, Hosein and Mueller, Aaron and Chen, Hanjie",
    booktitle = "Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP",
    month = nov,
    year = "2024",
    address = "Miami, Florida, US",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.blackboxnlp-1.7",
    pages = "105--117"
}
```
