# Fine-tuning and evaluation of the BERT-based Natural Language Inference (NLI) classifier

## Overview
This directory contains the full code for training and evaluating the model that serves as baseline in our experiments. This model is simply a BERT encoder followed by a typical classification head that is fed with a pair of premise and hypothesis and predicts the NLI label (entailment, contradiction, neutral).

## Usage

### Training
In order to fine-tune a classifier for the NLI task using `bert-base-uncased` checkpoint from HuggingFace as the encoder module, you can run the following command:

```bash
python fine-tuning/expl-free/train.py \
    --encoder_checkpt bert-base-uncased \
    --encoder_max_len 128 \
    --output_dir path/to/output/dir \
    --per_device_batch_size 32 \
    --num_train_epochs 5 \
    --logging_steps 1000 \
    --num_train_samples -1 \
    --num_eval_samples -1 \
    --seed 123
```

### Inference
In order to use a fine-tuned BERT-based NLI classifier (stored in file *path/to/fine-tuned/bert-nli/model.pt*) for inference, you can run the following command:

```bash
python fine-tuning/expl-free/inference.py \
    --trained_model path/to/fine-tuned/bert-nli/model.pt \
    --encoder_checkpt bert-base-uncased \
    --encoder_max_len 128 \
    --output_dir path/to/output/dir \
    --batch_size 32 \
    --num_test_samples -1 \
```

:warning:**Note:**:warning: The value of `encoder_checkpt` parameter during inference must match the checkpoint that was used for fine-tuning the classifier.