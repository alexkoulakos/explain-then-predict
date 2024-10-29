import torch
import torch.nn as nn
import argparse
import random
import numpy as np
import transformers
import os
import logging
import time

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

from utils import *

ID2LABEL = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.n_gpus > 0:
        torch.cuda.manual_seed_all(args.seed)

def setup(args):
    init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)

def cleanup():
    destroy_process_group()

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs

def display_info(device, step, progress, loss, acc, mins_elapsed, secs_elapsed, premise, hypothesis, gold_label, pred_label, mode):
    assert mode in ['train', 'eval']

    logger.info(f"Device: {device}")
    logger.info(f"iter: {step} | progress: {progress:.2f}%")
    logger.info(f"avg. {mode} loss: {loss} | avg. {mode} acc: {acc} | time elapsed: {mins_elapsed} mins {secs_elapsed} secs")
    logger.info(f"PREMISE: {premise}")
    logger.info(f"HYPOTHESIS: {hypothesis}")
    logger.info(f"GOLD LABEL: {gold_label}")
    logger.info(f"PREDICTED LABEL: {pred_label}")

def train_epoch(model, tokenizer, epoch, dataloader, optimizer, criterion, args):
    model.train()

    if args.local_rank in [-1, 0]:
        logger.info(100 * '-')
        logger.info(f'Epoch {epoch} running ...')

    epoch_start_time = time.time()
    train_loss = 0
    train_acc = 0

    optimizer.zero_grad()

    set_seed(args)

    if args.use_distributed_training:
        dataloader.sampler.set_epoch(epoch)

    for step, batch in enumerate(dataloader, 1):
        encoder_input = tokenizer.encode(batch['premise'], batch['hypothesis'])

        input_ids = encoder_input['input_ids'].to(args.device)
        attention_mask = encoder_input['attention_mask'].to(args.device)

        if hasattr(model, 'module'):
            label_distributions = model.module(input_ids, attention_mask)
        else:
            label_distributions = model(input_ids, attention_mask)
        
        pred_labels = label_distributions.argmax(dim=-1)
        gold_labels = batch['label'].to(args.device)

        loss = criterion(label_distributions, gold_labels)
        loss = loss / args.gradient_accumulation_steps

        train_loss += loss
        train_acc += pred_labels.long().eq(gold_labels.long()).cpu().sum() / len(pred_labels)

        loss.backward()
        
        if step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        if step % args.logging_steps == 0 and args.local_rank in [-1, 0]:
            progress = 100 * (step/(len(dataloader)))
            mins, secs = epoch_time(epoch_start_time, time.time())

            display_info(
                args.device,
                step, 
                progress, 
                train_loss / step, 
                train_acc / step, 
                mins,
                secs, 
                batch['premise'][0],
                batch['hypothesis'][0],
                ID2LABEL[gold_labels[0].item()],
                ID2LABEL[pred_labels[0].item()],
                'train'
            )
        
    return train_loss / len(dataloader), train_acc / len(dataloader), epoch_start_time

# IMPORTANT: Vaidation happens only on GPU 0!
def evaluate_epoch(model, tokenizer, dataloader, criterion, args):
    model.eval()

    eval_start_time = time.time()

    eval_loss = 0
    eval_acc = 0

    with torch.no_grad():
        for step, batch in enumerate(dataloader, 1):
            encoder_input = tokenizer.encode(batch['premise'], batch['hypothesis'])

            input_ids = encoder_input['input_ids'].to(args.device)
            attention_mask = encoder_input['attention_mask'].to(args.device)

            if hasattr(model, 'module'):
                label_distributions = model.module(input_ids, attention_mask)
            else:
                label_distributions = model(input_ids, attention_mask)
            
            pred_labels = label_distributions.argmax(dim=-1)
            gold_labels = batch['label'].to(args.device)

            loss = criterion(label_distributions, gold_labels)
            # loss = loss / args.gradient_accumulation_steps

            eval_loss += loss
            eval_acc += pred_labels.long().eq(gold_labels.long()).cpu().sum() / len(pred_labels)
            
            if step % args.logging_steps == 0 and args.local_rank in [-1, 0]:
                progress = 100 * (step/(len(dataloader)))
                mins, secs = epoch_time(eval_start_time, time.time())

                display_info(
                    args.device,
                    step, 
                    progress, 
                    eval_loss / (3 * step), 
                    eval_acc / (3 * step), 
                    mins,
                    secs, 
                    batch['premise'][0],
                    batch['hypothesis'][0],
                    ID2LABEL[gold_labels[0].item()],
                    ID2LABEL[pred_labels[0].item()],
                    'eval'
                )
        
    return eval_loss / len(dataloader), eval_acc / len(dataloader), eval_start_time

def main():
    parser = argparse.ArgumentParser()

    # Required params
    parser.add_argument("--train_data_file", type=str, required=True,
                        help="Path to training data csv file.")
    parser.add_argument("--eval_data_file", type=str, required=True,
                        help="Path to validation data csv file.")
    
    # Non-required params
    parser.add_argument("--encoder_checkpoint", default="bert-base-uncased", type=str, 
                        help="Encoder checkpoint for weights initialization.")
    
    parser.add_argument("--encoder_max_length", type=int, default=256, 
                        help="Max number of tokens the encoder can process.")
    
    parser.add_argument("--num_train_epochs", default=3, type=int, 
                        help="Number of training epochs.")
    parser.add_argument("--per_device_batch_size", default=16, type=int,
                        help="Batch size per GPU for training and validation.")
    
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of update steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--logging_steps", type=int, default=1000, 
                        help="Number of update steps between two logs.")
    
    parser.add_argument('--seed', type=int, default=123,
                        help="Random seed for initialization.")
    
    parser.add_argument("--num_train_samples", type=int, default=-1, 
                        help="Number of samples to use for training (-1 corresponds to the entire train set).")
    parser.add_argument('--num_eval_samples', type=int, default=-1,
                        help="Number of samples to use for evaluation (-1 corresponds to the entire validation set).")
    
    parser.add_argument("--use_distributed_training", action="store_true",
                        help="Whether to use distributed setup (multiple GPUs) for training.")
    
    args = parser.parse_args()
    
    n_gpus = torch.cuda.device_count()

    # Set up single-GPU/distributed training
    if args.use_distributed_training:
        assert n_gpus > 1, "Distributed training with torchrun requires at least 2 GPUs!"
        
        args.local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", args.local_rank)
        args.n_gpus = n_gpus
        setup(args)
    else:
        args.local_rank = -1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device == torch.device('cuda'):
            args.n_gpus = 1
        else:
            args.n_gpus = 0
    
    args.device = device

    if args.train_data_file.split('.')[-1] != 'csv':
        raise ValueError("Training data file must be csv.")
    
    if args.eval_data_file.split('.')[-1] != 'csv':
        raise ValueError("Validation data file must be csv.")

    # Setup output dir (example: /train_results/bert-base-uncased_128___bs_32)
    encoder_substr = f"{args.encoder_checkpoint}_{args.encoder_max_length}"
    batch_size_substr = f"bs_{args.per_device_batch_size}"
    transformers_version = f"transformers:{transformers.__version__}"

    # args.output_dir = f"./train_results/{encoder_substr}___{batch_size_substr}"
    args.output_dir = f"./train_results/{transformers_version}/{args.encoder_checkpoint}/encoder_max_length_{args.encoder_max_length}__batch_size_{args.per_device_batch_size}__n_gpus_{args.n_gpus}"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Setup logging
    logging.basicConfig(filename=f"{args.output_dir}/output.log",
                        format = '%(asctime)s - %(levelname)s - %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)

    # Only process 0 performs logging in order to avoid output repetition (in case of distributed training)
    if args.local_rank in [-1, 0]:
        print(f"transformers: v. {transformers.__version__}")
        logger.info(f"transformers: v. {transformers.__version__}")
        logger.info(f"{args.n_gpus} GPU(s) available for training!")
        logger.info(f"Command line arguments: {args}")

    # Set seed
    set_seed(args)

    if args.use_distributed_training:
        # Barrier to make sure only process 0 in distributed training downloads model and tokenizer 
        if args.local_rank != 0:
            torch.distributed.barrier()
    
    # Load tokenizer for BERT-NLI task
    tokenizer = BertNLITokenizer(args.encoder_checkpoint, args.encoder_max_length)

    # Initialize Expl2Label model with the given checkpoint
    model = BertNLIModel(args.encoder_checkpoint)

    model.to(args.device)
    
    if args.use_distributed_training:
        # End of barrier
        if args.local_rank == 0:
            torch.distributed.barrier()
    
    if args.use_distributed_training:
        # Barrier to make sure only process 0 in distributed training downloads dataset
        if args.local_rank != 0:
            torch.distributed.barrier()

    train_dataset = EsnliDataset(args.train_data_file, rows=args.num_train_samples)
    eval_dataset = EsnliDataset(args.eval_data_file, rows=args.num_eval_samples)

    if args.use_distributed_training:
        # End of barrier
        if args.local_rank == 0:
            torch.distributed.barrier()
    
    sampler = DistributedSampler(train_dataset) if args.use_distributed_training else RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.per_device_batch_size, 
        pin_memory=True, 
        shuffle=False, 
        sampler=sampler
    )

    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=args.per_device_batch_size, 
        pin_memory=True, 
        shuffle=False, 
        sampler=SequentialSampler(eval_dataset)
    )

    start_epoch = 0 
    best_eval_acc = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-05)
    criterion = nn.CrossEntropyLoss(reduction='mean')

    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pt")

    if args.use_distributed_training:
        # Barrier to make sure that only process 0 checks for existing model checkpoints and performs logging
        if args.local_rank != 0:
            torch.distributed.barrier()

    if args.local_rank in [-1, 0]:
        # Train
        logger.info("*************** RUNNING TRAINING ***************")
        logger.info(f"examples: {len(train_dataset)}")
        logger.info(f"epochs: {args.num_train_epochs}")
        logger.info(f"batch size per GPU: {args.per_device_batch_size}")
        logger.info(f"gradient accumulation steps: {args.gradient_accumulation_steps}")

        if os.path.exists(checkpoint_path):
            ###### CHECK THAT AGAIN LATER! ######
            if args.device == torch.device("cpu"):
                checkpoint = torch.load(checkpoint_path, map_location=args.device)
            else:
                checkpoint = torch.load(checkpoint_path)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']

            logger.info(f"Resuming training from epoch {start_epoch + 1}")
    
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
    for epoch in range(start_epoch + 1, args.num_train_epochs + 1):
        _, _, _ = train_epoch(
            model, 
            tokenizer, 
            epoch, 
            train_dataloader,  
            optimizer, 
            criterion, 
            args
        )

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
            eval_loss, eval_acc, eval_start_time = evaluate_epoch(
                model, 
                tokenizer, 
                eval_dataloader, 
                criterion, 
                args
            )

            eval_mins, eval_secs = epoch_time(eval_start_time, time.time())

            output_dir = args.output_dir
            metrics_file = f"{output_dir}/results.out"
            model_file = f"{output_dir}/model.pt"
            checkpoint_file = f"{output_dir}/checkpoint.pt"

            with open(metrics_file, 'a') as f:
                f.write("----- Eval results -----\n")
                f.write(f"EPOCH {epoch}\n")
                f.write(f"\t eval_loss: {eval_loss}, eval_acc: {eval_acc}, time_elapsed: {eval_mins} mins & {eval_secs} secs.\n\n")
            
            logger.info(f'Completed epoch {epoch} | avg. eval loss {eval_loss} | avg. eval acc {eval_acc} | time elapsed: {eval_mins} mins {eval_secs} secs')
            
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'eval_loss': eval_loss,
                'eval_acc': eval_acc,
            }

            torch.save(checkpoint_dict, checkpoint_file)

            logger.info(f"Saving model checkpoint to {checkpoint_file}")
            
            if eval_acc > best_eval_acc:
                logger.info("New best model found")
                best_eval_acc = eval_acc
                torch.save({'model_state_dict': checkpoint_dict['model_state_dict']}, model_file)
                logger.info(f"Saving current best model to {model_file}")
        
        if args.use_distributed_training:
            if args.local_rank == 0:
                torch.distributed.barrier()

    if args.use_distributed_training:    
        cleanup()

if __name__ == '__main__':
    main()