# Seq2Seq Model

## Overview
The _seq2seq_ model variations generate a natural language explanation that describes the inference relationship between the input premise and hypothesis. E.g.

__premise:__ A Land Rover is being driven across a river.
__hypothesis:__ A vehicle is crossing a river.
__explanation:__ Land Rover is a vehicle.

## Training
```bash
python train.py \
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
    --num_eval_samples -1
```

## Inference