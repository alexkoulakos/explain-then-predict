#!/bin/bash -l

####################################
#                                  #
# Submit script: sbatch filename   #
#                                  #
####################################

#SBATCH --job-name="bert-nli-attack"   # Job name
#SBATCH --output=/users/pa21/ptzouv/akoulakos/bert-nli/slurm_outputs/bert-nli-attack-000000.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=/users/pa21/ptzouv/akoulakos/bert-nli/slurm_outputs/bert-nli-attack-000000.%j.err # Stderr (%j expands to jobId)

#SBATCH --ntasks=1     # Number of tasks(processes)
#SBATCH --nodes=1     # Number of nodes requested
#SBATCH --ntasks-per-node=1     # Tasks per node
#SBATCH --cpus-per-task=1     # Threads per task
#SBATCH --time=48:00:00   # walltime
#SBATCH --mem=56G   # memory per NODE
#SBATCH --partition=ml    # Partition
#SBATCH --gres=gpu:1

#SBATCH --account=pa210503

if [ x$SLURM_CPUS_PER_TASK == x ]; then
  export OMP_NUM_THREADS=1
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

## LOAD MODULES ##
module purge
module load gnu/8 cuda/10.1.168 tftorch/270-191
# module load gnu/8 cuda/10.1.168 pytorch/1.9.1

source /users/pa21/ptzouv/akoulakos/textattackvenv/bin/activate

export PYTHONPATH=/users/pa21/ptzouv/akoulakos/textattackvenv/lib/python3.8/site-packages:$PYTHONPATH

srun python3 /users/pa21/ptzouv/akoulakos/bert-nli/attack.py \
    --bert_nli_trained_model "/users/pa21/ptzouv/akoulakos/bert-nli/train_results/transformers:4.30.0/bert-base-uncased/encoder_max_length_128__batch_size_16__n_gpus_1/model.pt" \
    --generation_config "/users/pa21/ptzouv/akoulakos/attack/generation_configs/greedy_search.json" \
    --output_dir "/users/pa21/ptzouv/akoulakos/bert-nli/foo_results/TextFooler/bert-base-uncased/attack_premise/greedy_search/" \
    --min_cos_sim 0.5 0.6 0.7 0.75 0.8 \
    --max_candidates 50 \
    --target_sentence "premise" \
    --num_samples -1 \
    --seed 123

deactivate
