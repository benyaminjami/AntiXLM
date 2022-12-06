#!/bin/bash
#SBATCH --job-name=AntiXLM
#SBATCH --time=12:00:00
#SBATCH --qos=normal

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx6000
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --export=ALL

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=16
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(/opt/slurm/bin/scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export LD_LIBRARY_PATH=/pkgs/nccl_2.9.9-1+cuda11.3_x86_64/lib:LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/pkgs/cuda-11.3/lib64:LD_LIBRARY_PATH
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
module load cuda-11.3
source ~/.bashrc 
conda activate AntiXLM
echo "Env Loaded"
### the command to run
echo "hostname; sleep \$(( SLURM_PROCID * 5 )); nvidia-smi; export NGPU=16; 
python train.py \
--cuda True \
--exp_name unsupMT_agab \
--dump_path /h/benjami/AntiXLM/dumped/ \
--data_path /h/benjami/AntiXLM/data/ \
--lgs 'ab-ag' \
--ae_steps 'ab,ag' \
--bt_steps 'ab-ag-ab,ag-ab-ag' \
--word_shuffle 3 \
--word_dropout 0.1 \
--word_blank 0.1 \
--lambda_ae '0:1,100000:0.1,300000:0' \
--max_len 160,250 \
--encoder_only false \
--emb_dim 1024 \
--n_layers 6 \
--n_heads 8 \
--dropout 0.1 \
--attention_dropout 0.1 \
--gelu_activation true \
--tokens_per_batch 3000 \
--batch_size 128 \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
--epoch_size 51200 \
--eval_bleu true \
--beam_size 1 \
--stopping_criterion 'valid_ag-ab_mt_bleu,10' \
--validation_metrics 'valid_ag-ab_mt_bleu' \
--master_port 12340" > srun_worker.sh

srun --mem=64GB --cpus-per-task=16 bash srun_worker.sh 
wait


