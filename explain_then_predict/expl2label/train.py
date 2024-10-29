import torch
import torch.nn as nn
import transformers
import argparse
import os
import logging
import time
import sys

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

sys.path.append("../")

from model import Expl2LabelModel
from tokenizer import Expl2LabelTokenizer
from helpers import *
from dataset import EsnliDataset

ID2LABEL = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

def train_epoch(
        model: Expl2LabelModel, 
        tokenizer: Expl2LabelTokenizer, 
        epoch: int, 
        dataloader: DataLoader, 
        optimizer, 
        criterion, 
        args
    ) -> dict:
    """
    Train `model` for an epoch using training `data`.
    Returns a `dict` with the calculated cross-entropy loss and accuracy.
    """
    model.train()

    if args.local_rank in [-1, 0]:
        args.logger.info(100 * '-')
        args.logger.info(f'Epoch {epoch} running ...\n')

    train_loss = 0
    train_acc = 0

    optimizer.zero_grad()

    set_seed(args)

    if args.use_distributed_training:
        dataloader.sampler.set_epoch(epoch)

    for step, batch in enumerate(dataloader, 1):
        encoder_input = tokenizer(batch['explanation_1'])

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
            hours, mins = compute_time(args.start_time, time.time())

            display_training_progress(
                args.logger,
                step, 
                progress, 
                train_loss/step, 
                train_acc/step, 
                hours,
                mins, 
                batch['explanation_1'][0],
                ID2LABEL[gold_labels[0].item()],
                ID2LABEL[pred_labels[0].item()],
            )

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return {
        'loss': train_loss,
        'acc': train_acc
    }

# IMPORTANT: Vaidation happens only on GPU 0!
def evaluate_epoch(
        model: Expl2LabelModel, 
        tokenizer: Expl2LabelTokenizer, 
        dataloader: DataLoader, 
        criterion, 
        args
    ) -> dict:
    """
    Evaluate model using validation `data`.
    Returns a `dict` with the calculated cross-entropy loss and accuracy.
    """
    model.eval()

    validation_loss = 0
    validation_acc = 0

    with torch.no_grad():
        for step, batch in enumerate(dataloader, 1):
            gold_labels = batch['label'].to(args.device)

            for index in range(1, 4):
                encoder_input = tokenizer(batch[f"explanation_{index}"])

                input_ids = encoder_input['input_ids'].to(args.device)
                attention_mask = encoder_input['attention_mask'].to(args.device)

                if hasattr(model, 'module'):
                    label_distributions = model.module(input_ids, attention_mask)
                else:
                    label_distributions = model(input_ids, attention_mask)
                
                pred_labels = label_distributions.argmax(dim=-1)
                gold_labels = batch['label'].to(args.device)

                loss = criterion(label_distributions, gold_labels)

                validation_loss += loss
                validation_acc += pred_labels.long().eq(gold_labels.long()).cpu().sum() / len(pred_labels)
    
    validation_loss /= 3 * len(dataloader)
    validation_acc /= 3 * len(dataloader)

    return {
        'loss': validation_loss,
        'acc': validation_acc
    }

def main():
    parser = argparse.ArgumentParser()

    # Required params
    parser.add_argument("--encoder_checkpt", required=True, type=str, 
                        help="Encoder checkpoint for weights initialization.")
    
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data csv file.")
    parser.add_argument("--validation_data", type=str, required=True,
                        help="Path to validation data csv file.")
    
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store output files (checkpoints, trained_model, log file).")
    
    # Non-required params
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
                        help="Number of samples to use for evaluation (-1 corresponds to the entire validation set).")
    
    parser.add_argument("--use_distributed_training", action="store_true",
                        help="Whether to use distributed setup (multiple GPUs) for training.")
    
    args = parser.parse_args()

    if args.train_data.split('.')[-1] != 'csv':
        raise ValueError("Training data file must be csv.")
    
    if args.validation_data.split('.')[-1] != 'csv':
        raise ValueError("Validation data file must be csv.")
    
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

    # Set up output files and sub-directories
    if not args.output_dir.endswith('/'):
        args.output_dir += "/"

    args.checkpt_dir = args.output_dir + "checkpoints/"
    args.model_config_dir = args.output_dir + "model_config/"
    args.results_dir = args.output_dir + "output_files/"

    checkpt_file = args.checkpt_dir + "checkpoint.pt"
    logging_file = args.results_dir + "output.log"
    train_synopsis_file = args.results_dir + "train_synopsis.out"
    
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
            print(f"{args.n_gpus} GPU(s) available for training!\n")
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
    
    # Load tokenizer for ExplanationToLabel task
    tokenizer = Expl2LabelTokenizer(args.encoder_checkpt, args.encoder_max_len)

    # Initialize Expl2Label model with the given checkpoint
    model = Expl2LabelModel(args.encoder_checkpt)

    model.to(args.device)
    
    if args.use_distributed_training:
        # End of barrier
        if args.local_rank == 0:
            torch.distributed.barrier()
    
    if args.use_distributed_training:
        # Barrier to make sure only process 0 in distributed training downloads dataset
        if args.local_rank != 0:
            torch.distributed.barrier()
    
    train_data = EsnliDataset(args.train_data, rows=args.num_train_samples)
    validation_data = EsnliDataset(args.validation_data, rows=args.num_eval_samples)

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
    best_validation_acc = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-05)
    criterion = nn.CrossEntropyLoss(reduction='mean')

    if args.use_distributed_training:
        # Barrier to make sure that only process 0 checks for existing model checkpoints and performs logging
        if args.local_rank != 0:
            torch.distributed.barrier()

    if args.local_rank in [-1, 0]:
        # Train
        args.logger.info("======================================= LAUNCHING TRAINING =======================================")
        args.logger.info(f"Train samples: {len(train_data)}")
        args.logger.info(f"Validation samples: {len(validation_data)}")
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
    
    with open(train_synopsis_file, 'a') as f:
        f.write("-------------- Training synopsis --------------\n\n")

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
            val_results_dict = evaluate_epoch(
                model, 
                tokenizer, 
                validation_dataloader, 
                criterion, 
                args
            )

            val_loss, val_acc = val_results_dict['loss'], val_results_dict['acc']

            time_elapsed_dict = compute_time(args.start_time, time.time())

            elapsed_hours, elapsed_mins = time_elapsed_dict['elapsed_hours'], time_elapsed_dict['elapsed_mins']

            args.logger.info(f"Completed epoch {epoch} in {elapsed_hours} hours & {elapsed_mins} mins\n")
            args.logger.info(f"\tvalidation loss {val_loss} | validation acc {val_acc}\n")

            with open(train_synopsis_file, 'a') as f:
                f.write(f"EPOCH {epoch} (ran in {elapsed_hours} hours & {elapsed_mins} mins)\n")
                f.write(f"\ttrain loss: {train_loss}, train acc: {train_acc}\n")
                f.write(f"\tvalidation loss: {val_loss}, validation acc: {val_acc}\n")
            
            checkpt_dict = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'validation_loss': val_loss,
                'validation_acc': val_acc,
            }

            torch.save(checkpt_dict, checkpt_file)

            args.logger.info(f"SAVING MODEL CHECKPOINTS TO: {checkpt_file}\n")
            
            if val_acc > best_validation_acc:
                args.logger.info("NEW BEST MODEL FOUND\n")
                args.logger.info(f"SAVING CURRENT BEST MODEL TO: {args.model_config_dir}\n")

                model_dict = {
                    'model_state_dict': checkpt_dict['model_state_dict'],
                    'args': args
                }

                torch.save(model_dict, f"{args.output_dir}/model.pt")

                # model.save_pretrained(args.model_config_dir)

                best_validation_acc = val_acc
        
        if args.use_distributed_training:
            if args.local_rank == 0:
                torch.distributed.barrier()

    if args.use_distributed_training:    
        cleanup()

if __name__ == '__main__':
    main()