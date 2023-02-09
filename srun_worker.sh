hostname; sleep $(( SLURM_PROCID * 10 )); nvidia-smi;
case a40 in
a40)
tokens=7000
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
echo a40;
torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=4 --max_restarts=1 --rdzv_backend=c10d --rdzv_endpoint=172.17.8.$(expr ${MASTER_ADDR:3:3} - 0) train.py --cuda True --exp_name unsupMT_agab --dump_path /checkpoint/${USER}/${SLURM_JOB_ID} --data_path /h/benjami/AntiXLM/data/ --lgs 'ab-ag' --ae_steps 'ab,ag' --bt_steps 'ab-ag-ab,ag-ab-ag' --mt_steps 'ag-ab,ab-ag' --mt_steps_ratio 25 --mt_steps_warmup 0 --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 --lambda_ae '0:1,100000:0.1,300000:0' --max_len 160,250 --encoder_only false --emb_dim 1024 --n_enc_layers 4 --n_dec_layers 6 --n_heads 8 --dropout 0.1 --accumulate_gradients 1 --attention_dropout 0.1 --gelu_activation true --tokens_per_batch $tokens --batch_size 512 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.001 --epoch_size 125000 --save_periodic 60000 --eval_bleu true --beam_size 10 --stopping_criterion 'valid_ag-ab_mt_bleu,10' --validation_metrics 'valid_ag-ab_mt_bleu' --master_port 12340
