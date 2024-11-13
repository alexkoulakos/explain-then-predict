# Expl2Label Model

## Overview
The _expl2label_ model predicts the NLI classification label (entailment, contradiction, neutral) based on the input explanation that describes the inference relationship between the premise and hypothesis. E.g.

__explanation:__ Land Rover is a vehicle. [corresponds to the pair (premise, hypothesis) = (A Land Rover is being driven across a river, A vehicle is crossing a river)]

__label__: entailment

## Usage

### Training
In order to fine-tune an _expl2label_ model using `bert-base-uncased` checkpoint from HuggingFace as the encoder module, you can run the following command:

```bash
python models/expl-based/expl2label/train.py \
    --encoder_checkpt bert-base-uncased \
    --output_dir path/to/output/dir \
    --per_device_batch_size 32 \
    --num_train_epochs 5 \
    --logging_steps 1000 \
    --num_train_samples -1 \
    --num_eval_samples -1 \
    --seed 123
```

### Inference
In order to use a fine-tuned *expl2label* model (stored in file *path/to/expl2label/fine-tuned/model.pt*) for inference, you can run the following command:

```bash
python models/expl-based/expl2label/inference.py \
    --trained_model path/to/expl2label/fine-tuned/model.pt \
    --encoder_checkpt bert-base-uncased \
    --output_dir path/to/output/dir \
    --batch_size 32 \
    --num_test_samples -1 \
```

:warning:**Note:**:warning: The value of `encoder_checkpt` parameter during inference must match the checkpoint that was used for fine-tuning the *expl2label* model.