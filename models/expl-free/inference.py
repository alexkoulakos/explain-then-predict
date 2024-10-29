import torch
import argparse
import logging
import os
import transformers
import random
import numpy as np
import csv

from torch.utils.data import DataLoader, SequentialSampler

from utils import *

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required params
    parser.add_argument("--trained_model_file", type=str, required=True, 
                        help="Path to fine-tuned model file.")
    parser.add_argument("--data_file", default=None, type=str, required=True, 
                        help="Path to test data csv file.")

    # Non-required params
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for testing.")
    parser.add_argument("--num_samples", type=int, default=-1, 
                        help="Number of samples to use for testing (-1 corresponds to the entire test set).")
    
    parser.add_argument('--seed', type=int, default=123,
                        help="Random seed for initialization.")

    args = parser.parse_args()

    """
    All trained models are located in directories of the form `./train_results/{encoder_checkpoint}/encoder_max_length_{encoder_max_length}__batch_size_{batch_size}__n_gpus_{n_gpus}/model.pt`.
    A valid directory example is `./train_results/bert-base-uncased/encoder_max_length_128__batch_size_8__n_gpus_1/model.pt`
    It is clear that this is a convenient format, because using python built-in split() method, we can extract
    all the information about the encoder-decoder checkpoints and their corresponding max_lengths and thus at inference
    time we can load the trained encoder-decoder model automatically, with no hard-coding.
    """
    args.encoder_checkpoint =  args.trained_model_file.split('/')[-3]
    other_params = args.trained_model_file.split('/')[-2]
    args.encoder_max_length = int(other_params.split('__')[0].split('_')[-1])

    args.output_dir = f"./inference_results/transformers:{transformers.__version__}/{args.encoder_checkpoint}/{other_params}"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data_file.split('.')[-1] != 'csv':
        raise ValueError("Test data file must be csv.")
    
    # Setup logging
    logging.basicConfig(filename=f"{args.output_dir}/output.log",
                        format = '%(asctime)s - %(levelname)s - %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    
    logger.info(f"transformers: v. {transformers.__version__}")

    print(f"Command line arguments")
    logger.info(f"Command line arguments")
    for name in vars(args):
        value = vars(args)[name]

        print(f"{name}: {value}")
        logger.info(f"{name}: {value}")
    
    # Set seed
    set_seed(args)

    # Load tokenizer for BERT-NLI task
    tokenizer = BertNLITokenizer(args.encoder_checkpoint, args.encoder_max_length)

    # Load pretrained model
    model = BertNLIModel(args.encoder_checkpoint)

    model.to(args.device)

    if args.device == torch.device('cpu'):
        model.load_state_dict(torch.load(args.trained_model_file, map_location=args.device)['model_state_dict'])
    else:
        model.load_state_dict(torch.load(args.trained_model_file)['model_state_dict'])

    test_dataset = EsnliDataset(args.data_file, rows=args.num_samples)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        pin_memory=True, 
        shuffle=False, 
        sampler=SequentialSampler(test_dataset)
    )

    # Evaluate model on e-SNLI test data
    model.eval()

    ID2LABEL = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    headers = ['premise', 'hypothesis', 'gold_label', 'pred_label']

    # if os.path.exists('./results.csv'):
        # os.remove('./results.csv')
    
    metrics_file_path = os.path.join(args.output_dir, "metrics.txt")
    csv_file_path = os.path.join(args.output_dir, "results.csv")

    csv_file = open(csv_file_path, 'x')
    writer = csv.writer(csv_file)
    writer.writerow(headers)
    
    test_acc = 0

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader, 1):
            encoder_input = tokenizer.encode(batch['premise'], batch['hypothesis'])

            input_ids = encoder_input['input_ids'].to(args.device)
            attention_mask = encoder_input['attention_mask'].to(args.device)
            
            label_distributions = model(input_ids, attention_mask)
            pred_labels = label_distributions.argmax(dim=-1)
            gold_labels = batch['label'].to(args.device)
            
            test_acc += pred_labels.long().eq(gold_labels.long()).cpu().sum() / len(pred_labels)

            for i in range(len(pred_labels)):
                row = []

                row.append(batch['premise'])
                row.append(batch['hypothesis'])
                row.append(ID2LABEL[gold_labels[i].item()])
                row.append(ID2LABEL[pred_labels[i].item()])

                writer.writerow(row)

    csv_file.close()

    acc = test_acc / len(test_dataloader)

    with open(metrics_file_path, 'w') as f:
        f.write(f'Test acc: {acc:.4f}\n')

    logger.info(f'Test acc: {acc}')