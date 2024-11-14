# Fine-tuning and evaluation of the Seq2Seq Model

## Overview
The _seq2seq_ model variations generate a natural language explanation that describes the inference relationship between the input premise and hypothesis. E.g.

__premise:__ A Land Rover is being driven across a river.

__hypothesis:__ A vehicle is crossing a river.

__explanation:__ Land Rover is a vehicle.

## Example usage

### Training
In order to fine-tune a _seq2seq_ model using `bert-base-uncased` checkpoint from HuggingFace as the encoder module and the `gpt2` checkpoint as the decoder module, you can run the following command:

```bash
python fine-tuning/expl-based/seq2seq/train.py \
    --encoder_checkpt bert-base-uncased \
    --encoder_max_len 128 \
    --decoder_checkpt gpt2 \
    --decoder_max_len 64 \
    --text_generation_strategy greedy_search \
    --remove_punctuation \
    --validation_metric meteor \
    --per_device_batch_size 32 \
    --num_train_epochs 5 \
    --output_dir path/to/output/dir \
    --logging_steps 1000 \
    --num_train_samples -1 \
    --num_eval_samples -1 \
    --seed 123
```

### Inference
In order to use a fine-tuned *seq2seq* model (stored in directory _path/to/seq2seq/fine-tuned/model/dir_) for inference, you can run the following command:

```bash
python fine-tuning/expl-based/seq2seq/inference.py \
    --trained_model path/to/seq2seq/fine-tuned/model/dir \
    --text_generation_strategy greedy_search \
    --output_dir path/to/output/dir \
    --batch_size 32 \
    --num_test_samples -1 \
```

:warning:**Note:**:warning: In contrast to the Expl2Label setup, there is no need to specify during inference the encoder checkpoint as it is automatically inferred from the `config.json` file inside the *path/to/seq2seq/fine-tuned/model/dir* directory.