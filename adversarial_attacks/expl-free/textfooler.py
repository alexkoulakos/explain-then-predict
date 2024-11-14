import textattack
import torch
import argparse
import os
import sys
import logging
import warnings

# Add `utils.py` to interpreter path
current_file = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_file)))

from typing import List

from textattack.attacker import Attacker
from textattack.attack_args import AttackArgs

from textattack.attack_recipes import TextFoolerJin2019

from textattack.transformations.word_swaps import WordSwapEmbedding
from textattack.constraints.semantics import WordEmbeddingDistance

from model_wrapper import BertNLIModelWrapper
from model import BertNLIModel
from tokenizer import BertNLITokenizer
from utils import *

# Suppress all kinds of warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def construct_attacks(args) -> List[textattack.Attack]:
    attacks = []

    for min_cos_sim in args.min_cos_sim:
        # Bind TextFooler attack recipe to model wrapper
        attack = TextFoolerJin2019.build(model_wrapper)

        # Create WordSwapEmbedding transformation according to `max_candidates` parameter.
        attack.transformation = WordSwapEmbedding(max_candidates=args.max_candidates)

        # Create WordEmbeddingDistance constraint according to `min_cos_sim` parameter.
        constraint = WordEmbeddingDistance(min_cos_sim=min_cos_sim)

        for i in range(len(attack.constraints)):
            if isinstance(attack.constraints[i], WordEmbeddingDistance):
                attack.constraints[i] = constraint
        
        attacks.append(add_input_column_modification(attack, args.column_to_ignore))
    
    return attacks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required params
    parser.add_argument("--trained_model_file", type=str, required=True, 
                        help="Path to fine-tuned model file.")
    
    parser.add_argument("--encoder_checkpt", required=True, type=str, 
                        help="Encoder checkpoint for the fine-tuned encoder part.")
    
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store output files (attack results synopsis and analysis).")

    # Non-required params
    parser.add_argument("--encoder_max_len", type=int, default=128, 
                        help="Max number of tokens the encoder can process.")
    
    parser.add_argument("--min_cos_sim", nargs="+", type=float, default=0.5,
                        help="List of cosine similarity thresholds to use in each run.")
    parser.add_argument("--max_candidates", type=int, default=50,
                        help="Max number of candidates (same for all runs).")
    
    parser.add_argument("--target_sentence", type=str, default="premise",
                        help="Input sentence to be perturbed (premise or hypothesis)")
    parser.add_argument("--num_samples", type=int, default=-1, 
                        help="Number of samples to attack (-1 corresponds to the entire test set)")
    
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed for initialization")

    args = parser.parse_args()

    assert args.target_sentence in ['premise', 'hypothesis'], "Parameter `target_sentence` can only have the values 'premise' or 'hypothesis'"

    if args.target_sentence == 'premise':
        args.column_to_ignore = 'hypothesis'
    else:
        args.column_to_ignore = 'premise'
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up output directory and logging file
    os.makedirs(args.output_dir, exist_ok=True)

    logging_file = os.path.join(args.output_dir, "output.log")

    # Setup logging
    args.logger = logging.getLogger(__name__)

    logging.basicConfig(filename=logging_file,
                        format = '%(asctime)s - %(levelname)s - %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    
    print(f"Command line arguments:")
    args.logger.info(f"Command line arguments:")
    for name in vars(args):
        value = vars(args)[name]

        print(f"\t{name}: {value}")
        args.logger.info(f"\t{name}: {value}")

    # Load pretrained model
    model = BertNLIModel(
        args.encoder_checkpt,
        args.device
    )

    model.to(args.device)

    if args.device == torch.device('cpu'):
        model.load_state_dict(torch.load(args.trained_model_file, map_location=args.device)['model_state_dict'])
    else:
        model.load_state_dict(torch.load(args.trained_model_file)['model_state_dict'])
    
    # Load tokenizer for NLI task
    tokenizer = BertNLITokenizer(
        args.encoder_checkpt, 
        args.encoder_max_len
    )

    # Load full model via its model wrapper
    model_wrapper = BertNLIModelWrapper(model, tokenizer)

    # Specify victim dataset
    dataset = load_dataset()

    # Create attacks, based on the different values of min_cos_sim
    attacks = construct_attacks(args)

    for attack, min_cos_sim in zip(attacks, args.min_cos_sim):
        # Setup sub-directory indicating the corresponding attack parameters
        attack_params_str = f"max_candidates_{args.max_candidates}__min_cos_sim_{min_cos_sim}"
        current_output_dir = os.path.join(args.output_dir, attack_params_str)
        os.makedirs(current_output_dir, exist_ok=True)

        # Setup output files
        csv_logging_file = os.path.join(current_output_dir, "attack_results.csv")
        analyzed_attack_results_file = os.path.join(current_output_dir, "results_analysis.txt")
        
        # Redirect stdout to the log file
        sys.stdout = open(logging_file, 'w')

        # Attack all test samples with CSV logging and checkpoint saved every 1000 intervals
        attack_args = AttackArgs(
            num_examples=args.num_samples,
            log_to_csv=csv_logging_file,
            checkpoint_interval=1000,
            checkpoint_dir=os.path.join(current_output_dir, "checkpoints"),
            disable_stdout=False,
            enable_advance_metrics=False,
            random_seed=args.seed
        )

        attacker = Attacker(attack, dataset, attack_args)
        attacker.attack_dataset()

        attack_results_df = pd.read_csv(csv_logging_file)

        with open(analyzed_attack_results_file, 'w') as f:
            for label in ['entailment', 'contradiction', 'neutral']:
                percent, percent_perturbed = analyze_attack_results(attack_results_df, label)

                if isinstance(percent, float):
                    f.write(f"Percentage of {label} labels that were successfully attacked: {percent:.2f}%\n")

                for k in percent_perturbed.keys():
                    f.write(f'\tPercentage of rows with "perturbed_output" = {k}: {percent_perturbed[k]:.2f}%\n')
                f.write("\n")