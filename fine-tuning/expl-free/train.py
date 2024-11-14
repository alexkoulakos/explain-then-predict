import torch
import argparse
import os
import logging
import time

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from model import BertNLIModel
from tokenizer import BertNLITokenizer
from dataset import SNLIDataset
from utils import *

def train_epoch(
        model: BertNLIModel, 
        tokenizer: BertNLITokenizer, 
        epoch: int, 
        dataloader: DataLoader, 
        optimizer: Adam, 
        criterion: CrossEntropyLoss, 
        args
    ):
    model.train()

    if args.local_rank in [-1, 0]:
        args.logger.info(100 * '-')
        args.logger.info(f'Epoch {epoch} running ...')

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
            time_elapsed_dict = compute_time(epoch_start_time, time.time())
            mins, secs = time_elapsed_dict['elapsed_mins'], time_elapsed_dict['elapsed_secs']

            display_training_progress(
                args.logger,
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

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return {
        'loss': train_loss,
        'acc': train_acc
    }    

# IMPORTANT: Validation happens only on GPU 0!
def validate(
        model: BertNLIModel, 
        tokenizer: BertNLITokenizer, 
        dataloader: DataLoader, 
        criterion: CrossEntropyLoss, 
        args
    ):
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
                time_elapsed_dict = compute_time(eval_start_time, time.time())
                mins, secs = time_elapsed_dict['elapsed_mins'], time_elapsed_dict['elapsed_secs']

                display_training_progress(
                    args.logger,
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
    
    eval_loss /= len(dataloader)
    eval_acc /= len(dataloader)

    return {
        'loss': eval_loss,
        'acc': eval_acc
    }
        
def main():
    parser = argparse.ArgumentParser()

    # Required param
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store output files (checkpoints, trained_model, log file).")
    
    # Non-required params
    parser.add_argument("--encoder_checkpt", default="bert-base-uncased", type=str, 
                        help="Encoder checkpoint for weights initialization.")
    
    parser.add_argument("--encoder_max_len", type=int, default=256, 
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
                        help="Number of samples to use for validation (-1 corresponds to the entire validation set).")
    
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

    # Set up output directory and files
    os.makedirs(args.output_dir, exist_ok=True)

    checkpt_file = os.path.join(args.output_dir, "checkpoint.pt")
    model_file = os.path.join(args.output_dir, "model.pt")
    logging_file = os.path.join(args.output_dir, "output.log")
    train_synopsis_file = os.path.join(args.output_dir, "train_synopsis.out")
    
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
    
    # Load tokenizer for the NLI task
    tokenizer = BertNLITokenizer(args.encoder_checkpt, args.encoder_max_len)

    # Initialize Expl2Label model with the given checkpoint
    model = BertNLIModel(args.encoder_checkpt, args.device)

    model.to(args.device)
    
    if args.use_distributed_training:
        # End of barrier
        if args.local_rank == 0:
            torch.distributed.barrier()
    
    if args.use_distributed_training:
        # Barrier to make sure only process 0 in distributed training downloads dataset
        if args.local_rank != 0:
            torch.distributed.barrier()

    train_data = SNLIDataset("train", rows=args.num_train_samples)
    eval_data = SNLIDataset("validation", rows=args.num_eval_samples)

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
        eval_data, 
        batch_size=args.per_device_batch_size, 
        pin_memory=True, 
        shuffle=False, 
        sampler=SequentialSampler(eval_data)
    )

    start_epoch = 0 
    best_val_acc = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-05)
    criterion = CrossEntropyLoss(reduction='mean')

    if args.use_distributed_training:
        # Barrier to make sure that only process 0 checks for existing model checkpoints and performs logging
        if args.local_rank != 0:
            torch.distributed.barrier()

    if args.local_rank in [-1, 0]:
        # Train
        args.logger.info("*************** LAUNCHING TRAINING ***************")
        args.logger.info(f"Train samples: {len(train_data)}")
        args.logger.info(f"Validation samples: {len(train_data)}")
        args.logger.info(f"Epochs: {args.num_train_epochs}")
        args.logger.info(f"Batch size per GPU: {args.per_device_batch_size}")
        args.logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")

        if os.path.exists(checkpt_file):
            if args.device == torch.device("cpu"):
                checkpoint = torch.load(checkpt_file, map_location=args.device)
            else:
                checkpoint = torch.load(checkpt_file)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']

            args.logger.info(f"Resuming training from epoch {start_epoch + 1}")
    
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
    for epoch in range(start_epoch + 1, args.num_train_epochs + 1):
        train_results_dict = train_epoch(
            model, 
            tokenizer, 
            epoch, 
            train_dataloader,  
            optimizer, 
            criterion, 
            args
        )

        train_loss, train_acc = train_results_dict['loss'], train_results_dict['acc']

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
                criterion, 
                args
            )

            val_loss, val_acc = val_results_dict['loss'], val_results_dict['acc']

            time_elapsed_dict = compute_time(args.start_time, time.time())

            elapsed_mins, elapsed_secs = time_elapsed_dict['elapsed_mins'], time_elapsed_dict['elapsed_secs']

            args.logger.info(f"Completed epoch {epoch} in {elapsed_mins} mins & {elapsed_secs} secs\n")
            args.logger.info(f"\tvalidation loss {val_loss} | validation acc {val_acc}\n")

            with open(train_synopsis_file, 'a') as f:
                f.write(f"EPOCH {epoch}\n")
                f.write(f"\ttrain loss: {train_loss}, train acc: {train_acc}\n")
                f.write(f"\tvalidation loss: {val_loss}, validation acc: {val_acc}\n")
                        
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }

            torch.save(checkpoint_dict, checkpt_file)

            args.logger.info(f"SAVING MODEL CHECKPOINTS TO: {checkpt_file}")
            
            if val_acc > best_val_acc:
                args.logger.info("NEW BEST MODEL FOUND\n")
                args.logger.info(f"SAVING CURRENT BEST MODEL TO: {model_file}")
                
                model_dict = {
                    'model_state_dict': checkpoint_dict['model_state_dict'],
                    'args': args
                }
                
                torch.save(model_dict, model_file)

                best_val_acc = val_acc
        
        if args.use_distributed_training:
            if args.local_rank == 0:
                torch.distributed.barrier()

    if args.use_distributed_training:    
        cleanup()

if __name__ == '__main__':
    main()