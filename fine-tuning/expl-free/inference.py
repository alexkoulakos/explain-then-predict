import torch
import argparse
import logging
import os
import csv

from torch.utils.data import DataLoader, SequentialSampler

from model import BertNLIModel
from tokenizer import BertNLITokenizer
from dataset import SNLIDataset
from utils import *

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required params
    parser.add_argument("--trained_model_file", type=str, required=True, 
                        help="Path to fine-tuned model file.")
    
    parser.add_argument("--encoder_checkpt", required=True, type=str, 
                        help="Encoder checkpoint for the fine-tuned encoder part.")
    
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store output files. NOTE: Please, provide an ABSOLUTE path!")

    # Non-required params
    parser.add_argument("--encoder_max_len", type=int, default=128, 
                        help="Max number of tokens the encoder can process.")
    
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for testing.")
    parser.add_argument("--num_test_samples", type=int, default=-1, 
                        help="Number of samples to use for testing (-1 corresponds to the entire test set).")
    
    parser.add_argument('--seed', type=int, default=123,
                        help="Random seed for initialization.")

    args = parser.parse_args()

    # Set up output directory and files
    os.makedirs(args.output_dir, exist_ok=True)

    scores_file = os.path.join(args.output_dir, "scores.out")
    predictions_file = os.path.join(args.output_dir, "predictions.csv")
    logging_file = os.path.join(args.output_dir, "output.log")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup logging
    args.logger = logging.getLogger(__name__)

    logging.basicConfig(filename=logging_file,
                        format = '%(asctime)s - %(levelname)s - %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    
    print(f"Command line arguments")
    args.logger.info(f"Command line arguments")
    for name in vars(args):
        value = vars(args)[name]

        print(f"\t{name}: {value}")
        args.logger.info(f"\t{name}: {value}")
    
    # Set seed
    set_seed(args)

    # Load tokenizer for BERT-NLI task
    tokenizer = BertNLITokenizer(args.encoder_checkpt, args.encoder_max_len)

    # Load pretrained model
    model = BertNLIModel(args.encoder_checkpt, args.device)

    model.to(args.device)

    if args.device == torch.device('cpu'):
        model.load_state_dict(torch.load(args.trained_model_file, map_location=args.device)['model_state_dict'])
    else:
        model.load_state_dict(torch.load(args.trained_model_file)['model_state_dict'])

    test_data = SNLIDataset("test", rows=args.num_test_samples)
    test_dataloader = DataLoader(
        test_data, 
        batch_size=args.batch_size, 
        pin_memory=True, 
        shuffle=False, 
        sampler=SequentialSampler(test_data)
    )

    # Evaluate model on e-SNLI test data
    model.eval()

    headers = ['premise', 'hypothesis', 'gold_label', 'pred_label']

    csv_file = open(predictions_file, mode='x', newline='', encoding='utf-8')
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

                row.append(batch['premise'][i])
                row.append(batch['hypothesis'][i])
                row.append(ID2LABEL[gold_labels[i].item()])
                row.append(ID2LABEL[pred_labels[i].item()])

                writer.writerow(row)

    csv_file.close()

    acc = test_acc / len(test_dataloader)

    with open(scores_file, 'w') as f:
        f.write(f'Test acc: {acc:.4f}\n')

    args.logger.info(f'Test acc: {acc}')