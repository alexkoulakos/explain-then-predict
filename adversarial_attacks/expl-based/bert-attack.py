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

from transformers import GenerationConfig, AutoTokenizer

from textattack.attacker import Attacker
from textattack.attack_args import AttackArgs

from textattack.transformations.word_swaps import WordSwapMaskedLM
from textattack.attack_recipes import BERTAttackLi2020

from model_wrapper import ExplainThenPredictModelWrapper
from model import ExplainThenPredictModel
from utils import *
from generation_config import GENERATION_CONFIG

# Suppress all kinds of warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def construct_attacks(args) -> List[textattack.Attack]:
    attacks = []

    for max_candidates in args.max_candidates:
        # Bind BERT-attack attack recipe to model wrapper
        attack = BERTAttackLi2020.build(model_wrapper)

        # Create WordSwapMaskedLM transformation according to `max_candidates` parameter.
        attack.transformation = WordSwapMaskedLM(method="bert-attack", max_candidates=max_candidates)
        
        attacks.append(add_input_column_modification(attack, args.column_to_ignore))
    
    return attacks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required params
    parser.add_argument("--seq2seq_model", type=str, required=True, 
                        help="Path to the fine-tuned encoder-decoder model directory.")
    parser.add_argument("--expl2label_model", type=str, required=True, 
                        help="Path to the fine-tuned classifier model file.")
    
    parser.add_argument("--text_generation_strategy", required=True, type=str, 
                        help="Strategy to use for text generation during model validation. Strategy must be exactly one of: greedy_search, beam_search, top-k_sampling, top-p_sampling")
    
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store output files (attack results synopsis and analysis).")

    # Non-required params
    parser.add_argument("--encoder_max_len", type=int, default=128, 
                        help="Max number of tokens the seq2seq encoder can process.")
    
    parser.add_argument("--max_candidates", nargs="+", type=int, default=0.5,
                        help="List of K values to use in each run.")
    
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

    args.expl2label_encoder_checkpt = torch.load(args.expl2label_model, map_location=args.device)['args'].encoder_checkpt

    # Set up output directory and files
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

    # args.text_generation_strategy points to the corresponding entry of GENERATION_CONFIG
    config_dict = GENERATION_CONFIG[args.text_generation_strategy]

    # Instantiate GenerationConfig class based on config_dict
    args.generation_config = GenerationConfig(**config_dict)
    
    # Load model and tokenizer
    model = ExplainThenPredictModel(args).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model.seq2seq_model.config.encoder._name_or_path)

    # Load full model via its model wrapper
    model_wrapper = ExplainThenPredictModelWrapper(model, tokenizer)

    # Specify victim dataset
    dataset = load_dataset()

    # Create attacks, based on the different values of max_candidates (K)
    attacks = construct_attacks(args)

    for attack, max_candidates in zip(attacks, args.max_candidates):
        # Setup sub-directory indicating the corresponding attack parameters
        attack_params_str = f"max_candidates_{max_candidates}"
        current_output_dir = os.path.join(args.output_dir, attack_params_str)
        os.makedirs(current_output_dir, exist_ok=True)

        # Setup output files
        csv_logging_file = os.path.join(current_output_dir, "attack_results.csv")
        analyzed_attack_results_file = os.path.join(current_output_dir, "results_analysis.txt")

        # Redirect stdout to a log file
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