#!/bin/bash
#SBATCH --job-name=AntiXLM
#SBATCH --time=48:00:00
#SBATCH --qos=normal

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=140GB
#SBATCH --gres=gpu:4
#SBATCH --partition=a40#,rtx6000#,t4v2
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --export=ALL

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(/opt/slurm/bin/scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export LD_LIBRARY_PATH=/pkgs/nccl_2.9.9-1+cuda11.3_x86_64/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/pkgs/cuda-11.3/lib64:$LD_LIBRARY_PATH
echo "MASTER_ADDR="$MASTER_ADDR
# if [[ -f /checkpoint/benjami/${SLURM_JOB_ID}/unsupMT_agab/0/checkpoint.pth ]]; then
#     echo exist
# else
#     cp -r /checkpoint/benjami/9086550/unsupMT_agab/ /checkpoint/benjami/${SLURM_JOB_ID}/
# fi

# ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID} $PWD/checkpoint/${SLURM_JOB_ID}
### init virtual environment if needed
module load cuda-11.3 nccl_2.9.9-1+cuda11.3
source ~/.bashrc 
conda activate AntiXLM
echo "Env Loaded"
### the command to run
echo "hostname; sleep \$(( SLURM_PROCID * 10 )); nvidia-smi;
case $SLURM_JOB_PARTITION in
a40)
tokens=700
;;
rtx6000)
tokens=3000
;;
t4v2)
tokens=1000
;;
interactive)
echo don\'t know
tokens=2000
;;
esac;
echo $SLURM_JOB_PARTITION;
torchrun \
--nnodes=\$SLURM_JOB_NUM_NODES \
--nproc_per_node=4 \
--max_restarts=1 \
--rdzv_backend=c10d \
--rdzv_endpoint=172.17.8.\$(expr \${MASTER_ADDR:3:3} - 0) \
train.py \
--cuda True \
--exp_name unsupMT_agab \
--dump_path /checkpoint/\${USER}/\${SLURM_JOB_ID} \
--data_path /h/benjami/scrach/AntiXLM/data/ \
--lgs 'ab-ag' \
--ae_steps 'ab,ag' \
--bt_steps 'ab-ag-ab,ag-ab-ag' \
--mt_steps 'ag-ab,ab-ag' \
--mt_steps_ratio 25 \
--mt_steps_warmup 0 \
--word_shuffle 3 \
--word_dropout 0.15 \
--word_blank 0.15 \
--lambda_ae '0:1,100000:0.1,300000:0' \
--max_len 160,250 \
--encoder_only false \
--emb_dim 1024 \
--n_layers 8 \
--n_heads 8 \
--dropout 0.1 \
--accumulate_gradients 1 \
--attention_dropout 0.1 \
--gelu_activation true \
--tokens_per_batch \$tokens \
--batch_size 512 \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.001 \
--epoch_size 100000 \
--save_periodic 60000 \
--beam_size 10 \
--stopping_criterion 'valid_ag-ab_mt_bleu,10' \
--validation_metrics 'valid_ag-ab_mt_bleu' \
--master_port 12340
--debug_train True" > srun_worker.sh

srun --mem=140GB --cpus-per-task=8 bash srun_worker.sh 
wait
