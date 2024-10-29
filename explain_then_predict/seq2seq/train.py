import json
import torch
import transformers
import argparse
import os
import logging
import math
import time
import sys
import datasets

from transformers import (
    EncoderDecoderModel, 
    GenerationConfig, 
    AdamW, 
    get_linear_schedule_with_warmup
)
from torch.utils.data import (
    DataLoader, 
    SequentialSampler, 
    RandomSampler
)
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

sys.path.append("../")

from tokenizer import Seq2SeqTokenizer
from helpers import *
# from dataset import EsnliDataset

def train_epoch(
        model: EncoderDecoderModel, 
        tokenizer: Seq2SeqTokenizer, 
        epoch: int, 
        data: DataLoader, 
        optimizer, 
        scheduler, 
        args
    ):
    """
    Train `model` for an epoch using training `data`.
    Returns a `dict` with the calculated cross-entropy loss and perplexity.
    """
    model.train()

    if args.local_rank in [-1, 0]:
        args.logger.info(100 * '-')
        args.logger.info(f'Epoch {epoch} running ...\n')

    train_loss = 0
    train_ppl = 0

    model.zero_grad()

    set_seed(args)

    if args.use_distributed_training:
        data.sampler.set_epoch(epoch)

    for step, batch in enumerate(data, 1):
        encoder_input = tokenizer((batch['premise'], batch['hypothesis']))
        input_ids = encoder_input['input_ids'].to(args.device)
        attention_mask = encoder_input['attention_mask'].to(args.device)

        decoder_input = tokenizer(batch['explanation_1'], gpt2=True)
        decoder_input_ids = decoder_input['input_ids'].to(args.device)
        decoder_attention_mask = decoder_input['attention_mask'].to(args.device)

        labels = decoder_input_ids.clone()
        labels[decoder_attention_mask == 0] = -100
        labels = shift_tokens_left(labels)
        labels.to(args.device)

        if hasattr(model, 'module'):
            output = model.module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                return_dict=True
            )
        else:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                return_dict=True
            )

        logits, loss = output.logits, output.loss
        loss = loss / args.gradient_accumulation_steps
        ppl = math.exp(loss)

        train_loss += loss
        train_ppl += ppl

        loss.backward()
        
        if step % args.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            model.zero_grad()
        
        if step % args.logging_steps == 0 and args.local_rank in [-1, 0]:
            progress = 100 * (step / (len(data)))
            hours, mins = compute_time(args.start_time, time.time())

            display_training_progress(
                args.logger,
                step, 
                progress, 
                train_loss/step, 
                train_ppl/step, 
                hours,
                mins, 
                batch['premise'][0],
                batch['hypothesis'][0],
                batch['explanation_1'][0],
                tokenizer.decode(torch.argmax(logits, dim=2)[0])
            )
    
    train_loss /= len(data)
    train_ppl /= len(data)

    return {
        'loss': train_loss,
        'ppl': train_ppl
    }

def validate(
        model: EncoderDecoderModel, 
        tokenizer: Seq2SeqTokenizer, 
        data: DataLoader, 
        generation_config: GenerationConfig,
        args
    ) -> dict:
    """
    Evaluate model using validation `data`.
    Returns a `dict` with the calculated evaluation metrics.
    """
    model.eval()

    validation_loss = 0
    validation_ppl = 0

    predictions, expls_1, expls_2, expls_3 = [], [], [], []

    with torch.no_grad():
        for batch in data:
            encoder_input = tokenizer((batch['premise'], batch['hypothesis']))
            input_ids = encoder_input['input_ids'].to(args.device)
            attention_mask = encoder_input['attention_mask'].to(args.device)

            for index in range(1, 4):
                if args.remove_punctuation:
                    eval(f"expls_{index}").extend(remove_punctuation(batch[f'explanation_{index}']))
                else:
                    eval(f"expls_{index}").extend(batch[f'explanation_{index}'])

                decoder_input = tokenizer(batch[f'explanation_{index}'], gpt2=True)
                decoder_input_ids = decoder_input['input_ids'].to(args.device)
                decoder_attention_mask = decoder_input['attention_mask'].to(args.device)
                
                labels = decoder_input_ids.clone()
                labels[decoder_attention_mask == 0] = -100
                labels = shift_tokens_left(labels)
                labels.to(args.device)

                if hasattr(model, 'module'):
                    output = model.module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_attention_mask=decoder_attention_mask,
                        labels=labels,
                        return_dict=True
                    )
                else:
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_attention_mask=decoder_attention_mask,
                        labels=labels,
                        return_dict=True
                    )

                loss = output.loss
                ppl = math.exp(loss)

                validation_loss += loss
                validation_ppl += ppl
            
            # Generate explanation
            if hasattr(model, 'module'):
                output_ids = model.module.generate(
                    input_ids, 
                    attention_mask=attention_mask,
                    generation_config=generation_config
                )
            else:
                output_ids = model.generate(
                    input_ids, 
                    attention_mask=attention_mask,
                    generation_config=generation_config
                )
            
            """unprocessed_expls = tokenizer.batch_decode(output_ids)

            # Truncate to the first complete and meaningful sentence
            pred_expls = truncate_pred_expls(unprocessed_expls)"""

            pred_expls = tokenizer.batch_decode(output_ids)

            if args.remove_punctuation:
                predictions.extend(remove_punctuation(pred_expls))
            else:
                predictions.extend(pred_expls)
    
    references = [[expl_1, expl_2, expl_3] for expl_1, expl_2, expl_3 in zip(expls_1, expls_2, expls_3)]

    validation_loss /= 3 * len(data)
    validation_ppl /= 3 * len(data)

    metrics = compute_metrics(predictions, references)
    metrics['loss'] = validation_loss
    metrics['ppl'] = validation_ppl
    
    return metrics

def main():
    parser = argparse.ArgumentParser()

    # Required params
    parser.add_argument("--encoder_checkpt", required=True, type=str, 
                        help="Encoder checkpoint for weights initialization.")
    parser.add_argument("--decoder_checkpt", required=True, type=str, 
                        help="Decoder checkpoint for weights initialization.")
    
    parser.add_argument("--generation_config", required=True, type=str, 
                        help="Path to json file that contains the parameters for generation during model validation.")
    
    """parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data csv file.")
    parser.add_argument("--validation_data", type=str, required=True,
                        help="Path to validation data csv file.")"""
    
    parser.add_argument("--validation_metric", type=str, required=True,
                        help="Metric to use for model validation. Metric must be exactly one of: ppl, bleu, meteor, rouge_1, rouge_2, bert")
    
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store output files (checkpoints, trained_model, log file).")
    
    # Non-required params
    parser.add_argument("--encoder_max_len", type=int, default=128, 
                        help="Max number of tokens the encoder can process.")
    parser.add_argument("--decoder_max_len", type=int, default=64, 
                        help="Max number of tokens the decoder can process.")
    
    parser.add_argument("--num_train_epochs", default=5, type=int, 
                        help="Number of training epochs.")
    parser.add_argument("--per_device_batch_size", default=16, type=int,
                        help="Batch size per GPU for training and validation.")
    
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of update steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--logging_steps", type=int, default=1000, 
                        help="Number of update steps between two logs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    
    parser.add_argument('--seed', type=int, default=123,
                        help="Random seed for initialization.")
    
    parser.add_argument("--num_train_samples", type=int, default=-1, 
                        help="Number of samples to use for training (-1 corresponds to the entire train set).")
    parser.add_argument('--num_eval_samples', type=int, default=-1,
                        help="Number of samples to use for validation (-1 corresponds to the entire validation set).")
    
    parser.add_argument("--use_distributed_training", action="store_true",
                        help="Whether to use distributed setup (multiple GPUs) for training.")
    
    parser.add_argument("--remove_punctuation", action="store_true",
                        help="Whether to remove punctuation when calculating text generation metrics.")
    
    args = parser.parse_args()

    """if args.train_data.split('.')[-1] != 'csv':
        raise ValueError("Training data file must be in csv format.")
    
    if args.validation_data.split('.')[-1] != 'csv':
        raise ValueError("Validation data file must be in csv format.")"""
    
    if args.generation_config.split('.')[-1] != 'json':
        raise ValueError("Generation config file must be in json format.")
    
    assert args.validation_metric in ['ppl', 'bleu', 'meteor', 'rouge_1', 'rouge_2', 'bert'], "Please provide a valid metric for model validation. Valid metrics are: ppl, bleu, meteor, rouge_1, rouge_2, bert."

    n_gpus = torch.cuda.device_count()

    # Set up single-GPU or distributed training
    if args.use_distributed_training:
        assert n_gpus > 1, "Distributed training with torchrun requires at least 2 GPUs!"
        
        args.local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", args.local_rank)
        args.n_gpus = n_gpus
        setup(args)
    else:
        args.local_rank = -1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device == torch.device("cuda"):
            args.n_gpus = 1
        else:
            args.n_gpus = 0
    
    args.device = device

    # Set up output files and sub-directories
    if not args.output_dir.endswith('/'):
        args.output_dir += "/"

    args.checkpt_dir = args.output_dir + "checkpoints/"
    args.model_config_dir = args.output_dir + "model/"
    args.results_dir = args.output_dir + "output_files/"

    checkpt_file = args.checkpt_dir + "checkpoint.pt"
    logging_file = args.results_dir + "output.log"
    train_synopsis_file = args.results_dir + "synopsis.out"
    
    os.makedirs(args.checkpt_dir, exist_ok=True)
    os.makedirs(args.model_config_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Setup logging
    args.logger = logging.getLogger(__name__)

    logging.basicConfig(filename=logging_file,
                        format = '%(asctime)s - %(levelname)s - %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)

    # Only process 0 performs logging in order to avoid output repetition (in case of distributed training)
    if args.local_rank in [-1, 0]:
        if args.local_rank == 0:
            args.logger.info(f"{args.n_gpus} GPU(s) available for training!\n")
        
        print(f"transformers: v. {transformers.__version__}\n")
        args.logger.info(f"transformers: v. {transformers.__version__}\n")

        print(f"Command line arguments:")
        args.logger.info(f"Command line arguments:")
        
        for name in vars(args):
            value = vars(args)[name]

            print(f"\t{name}: {value}")
            args.logger.info(f"\t{name}: {value}")
    
    # Set seed
    set_seed(args)

    if args.use_distributed_training:
        # Barrier to make sure only process 0 in distributed training downloads model and tokenizer 
        if args.local_rank != 0:
            torch.distributed.barrier()
    
    # Load tokenizer for Seq2Seq task
    tokenizer = Seq2SeqTokenizer(
        args.encoder_checkpt,
        args.decoder_checkpt,
        args.encoder_max_len,
        args.decoder_max_len
    )

    # Initialize encoder-decoder model from HuggingFace model hub with the given checkpoints
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(args.encoder_checkpt, args.decoder_checkpt)
    
    model.decoder.config.use_cache = False

    # Decoder-related parameters
    model.config.decoder_start_token_id = tokenizer.decoder_tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.decoder_tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.decoder_tokenizer.pad_token_id
    model.config.transformers_version = transformers.__version__

    model.to(args.device)

    # Instantiate GenerationConfig class based on args.generation_config file    
    with open(args.generation_config) as f:
        config_dict = json.load(f)
    
    if config_dict['early_stopping'] in ['True', 'False']:
        config_dict['early_stopping'] = eval(config_dict['early_stopping'])
    
    config_dict['do_sample'] = eval(config_dict['do_sample'])
    
    if config_dict['penalty_alpha'] == 'None':
        config_dict['penalty_alpha'] = None

    config_dict['decoder_start_token_id'] = tokenizer.decoder_tokenizer.bos_token_id
    config_dict['eos_token_id'] = tokenizer.decoder_tokenizer.eos_token_id
    config_dict['pad_token_id'] = tokenizer.decoder_tokenizer.pad_token_id

    generation_config = GenerationConfig(**config_dict)

    if args.use_distributed_training:
        # End of barrier
        if args.local_rank == 0:
            torch.distributed.barrier()
    
    if args.use_distributed_training:
        # Barrier to make sure only process 0 in distributed training downloads dataset
        if args.local_rank != 0:
            torch.distributed.barrier()
    
    if args.num_train_samples > 0:
        train_data = datasets.load_dataset("esnli", split="train").filter(lambda example: example['label'] in [0, 1, 2]).select(range(args.num_train_samples))
    else:
        train_data = datasets.load_dataset("esnli", split="train").filter(lambda example: example['label'] in [0, 1, 2])
    
    if args.num_eval_samples > 0:
        validation_data = datasets.load_dataset("esnli", split="validation").filter(lambda example: example['label'] in [0, 1, 2]).select(range(args.num_eval_samples))
    else:
        validation_data = datasets.load_dataset("esnli", split="validation").filter(lambda example: example['label'] in [0, 1, 2])
    
    """train_data = EsnliDataset(args.train_data, rows=args.num_train_samples)
    validation_data = EsnliDataset(args.validation_data, rows=args.num_eval_samples)"""

    if args.use_distributed_training:
        # End of barrier
        if args.local_rank == 0:
            torch.distributed.barrier()
    
    sampler = DistributedSampler(train_data) if args.use_distributed_training else RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, 
        batch_size=args.per_device_batch_size, 
        pin_memory=True, 
        shuffle=False, 
        sampler=sampler
    )

    validation_dataloader = DataLoader(
        validation_data, 
        batch_size=args.per_device_batch_size, 
        pin_memory=True, 
        shuffle=False, 
        sampler=SequentialSampler(validation_data)
    )

    start_epoch = 0

    if args.validation_metric == 'ppl':
        best_score = float('inf')
    else:
        best_score = 0

    total_steps = args.num_train_epochs * len(train_dataloader) // args.gradient_accumulation_steps

    # Prepare optimizer and scheduler (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    if args.use_distributed_training:
        # Barrier to make sure that only process 0 checks for existig model checkpoints and performs logging
        if args.local_rank != 0:
            torch.distributed.barrier()

    if args.local_rank in [-1, 0]:
        # Train
        args.logger.info("======================================= LAUNCHING TRAINING =======================================")
        args.logger.info(f"Train samples: {len(train_data)}")
        args.logger.info(f"Validation samples: {len(validation_data)}")
        args.logger.info(f"Validation metric: {args.validation_metric}")
        args.logger.info(f"Epochs: {args.num_train_epochs}")
        args.logger.info(f"Batch size per GPU: {args.per_device_batch_size}")
        args.logger.info(f"Gradient Accumulation steps: {args.gradient_accumulation_steps}\n")

        if os.path.exists(checkpt_file):
            ###### CHECK THAT AGAIN LATER! ######
            if args.device == torch.device("cpu"):
                checkpoint = torch.load(checkpt_file, map_location=args.device)
            else:
                checkpoint = torch.load(checkpt_file)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']

            args.logger.info(f"=================================== RESUMING TRAINING FROM EPOCH {start_epoch + 1} ===================================")
    
    if args.use_distributed_training:
        # End of barrier
        if args.local_rank == 0:
            torch.distributed.barrier()
    
    if args.use_distributed_training:
        # Distributed training
        model = DistributedDataParallel(
            model, 
            device_ids=[args.local_rank], 
            find_unused_parameters=True
        )

    # Train for each epoch
    args.start_time = time.time()
    with open(train_synopsis_file, 'a') as f:
        f.write("---------------------------- Training synopsis ----------------------------\n\n")

    for epoch in range(start_epoch + 1, args.num_train_epochs + 1):
        train_results_dict = train_epoch(
            model, 
            tokenizer, 
            epoch, 
            train_dataloader,  
            optimizer, 
            scheduler,
            args
        )

        train_loss, train_ppl = train_results_dict['loss'], train_results_dict['ppl']

        """
        Evaluation happens only on GPU 0, because we need evaluation results gathered in a single device, 
        so that we can average them accurately. Consequently, all processes must wait for process 0 to finish
        with the evaluation procedure and afterwards they can safely continue running code. We will use barriers
        in order to achieve syncing among the processes.
        """
        if args.use_distributed_training:
            if args.local_rank != 0:
                torch.distributed.barrier()

        if args.local_rank in [-1, 0]:
            val_results_dict = validate(
                model, 
                tokenizer, 
                validation_dataloader, 
                generation_config,
                args
            )

            val_loss, val_ppl = val_results_dict['loss'], val_results_dict['ppl']
            bleu_score = val_results_dict['bleu_score']
            meteor_score = val_results_dict['meteor_score']
            rouge_1_score, rouge_2_score = val_results_dict['rouge_1_score'], val_results_dict['rouge_2_score']
            bert_score = val_results_dict['bert_score']

            time_elapsed_dict = compute_time(args.start_time, time.time())

            elapsed_hours, elapsed_mins = time_elapsed_dict['elapsed_hours'], time_elapsed_dict['elapsed_mins']

            args.logger.info(f"Completed epoch {epoch} in {elapsed_hours} hours & {elapsed_mins} mins\n")
            args.logger.info(f"\tvalidation loss {val_loss} | validation ppl {val_ppl}\n")
            args.logger.info(f"\tbleu: {bleu_score}, meteor: {meteor_score}, rouge-1: {rouge_1_score}, rouge-2: {rouge_2_score}, bert score: {bert_score}\n")

            with open(train_synopsis_file, 'a') as f:
                f.write(f"EPOCH {epoch} (ran in {elapsed_hours} hours & {elapsed_mins} mins)\n")
                f.write(f"\ttrain loss: {train_loss} | train ppl: {train_ppl}\n")
                f.write(f"\tvalidation loss: {val_loss} | validation ppl: {val_ppl}\n")
                f.write(f"\tbleu: {bleu_score} | meteor: {meteor_score} | rouge_1: {rouge_1_score} | rouge_2: {rouge_2_score} | bert score: {bert_score}\n")
            
            checkpt_dict = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'validation_loss': val_loss,
                'validation_ppl': val_ppl,
                'bleu_score': bleu_score,
                'meteor_score': meteor_score,
                'rouge_1_score': rouge_1_score,
                'rouge_2_score': rouge_2_score,
                'bert_score': bert_score
            }

            torch.save(checkpt_dict, checkpt_file)

            args.logger.info(f"SAVING MODEL CHECKPOINTS TO: {checkpt_file}\n")
            
            if args.validation_metric != 'ppl':
                current_score = val_results_dict[f'{args.validation_metric}_score']
            else:
                current_score = val_results_dict['ppl']
            
            if new_best_model_found(current_score, best_score, args.validation_metric):
                args.logger.info("NEW BEST MODEL FOUND")
                args.logger.info(f"SAVING CURRENT BEST MODEL TO: {args.model_config_dir}")

                model.save_pretrained(args.model_config_dir)

                best_score = current_score

        if args.use_distributed_training:
            if args.local_rank == 0:
                torch.distributed.barrier()

    if args.use_distributed_training:    
        cleanup()

if __name__ == '__main__':
    main()