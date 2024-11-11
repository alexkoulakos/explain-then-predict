# Explanation-based models

## Overview
This directory contains the full code for training and evaluating all the models that that follow the "Explain Then Predict" pipeline. According to this setup, the following apply:

1. First, a _seq2seq_ model is fed with a pair (premise, hypothesis) and outputs an explanation that describes the inference relationship between premise and hypothesis.
2. This explanation is fed to an _expl2label_ model which predicts the output label (entailment, contradiction, neutral) based on the input explanation.

__The above models are trained independently and they are joined together at inference only__.

For training details of the _expl2label_ or _seq2seq_ model, see the README file in the corresponding directory.

## Inference
In order to perform inference using the _seq2seq_ model (stored in directory _path/to/seq2seq/fine-tuned/model/dir_) followed by the _expl2label_ model (stored in file _path/to/expl2label/fine-tuned/model.pt_), you can use the below:

```bash
python models/expl-based/inference.py \
    --seq2seq_model path/to/seq2seq/fine-tuned/model/dir \
    --expl2label_model path/to/expl2label/fine-tuned/model.pt \
    --text_generation_strategy greedy_search \
    --output_dir path/to/output/dir \
    --remove_punctuation \
    --batch_size 32 \
    --num_test_samples -1 \
    --seed 123
```