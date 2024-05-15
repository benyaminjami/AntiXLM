hostname; sleep $(( SLURM_PROCID * 10 )); nvidia-smi;
case a40 in
a40)
tokens=2000
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
torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=3 --max_restarts=1 --rdzv_backend=c10d --rdzv_endpoint=172.17.8.$(expr ${MASTER_ADDR:3:3} - 0) train-T5.py --cuda True --exp_name unsupMT_agab --dump_path /checkpoint/${USER}/${SLURM_JOB_ID} --reload_checkpoint /checkpoint/benjami/10638900/unsupMT_agab/0/checkpoint.pth --data_path ~/scrach/T5-Data/ --lgs 'ab-ag' --ae_steps 'ab,ag' --bt_steps 'ab-ag-ab,ag-ab-ag' --mt_steps 'ag-ab,ab-ag' --mt_steps_ratio 1 --mt_steps_warmup 0 --word_shuffle 3 --word_dropout 0.15 --word_blank 0.15 --lambda_ae '0:1,50000:0.1,300000:0' --max_len 150,202 --min_len 90,20 --encoder_only false --accumulate_gradients 1 --tokens_per_batch $tokens --batch_size 10 --max_batch_size 15 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.001 --epoch_size 6000 --save_periodic 2000 --eval true --beam_size 5 --stopping_criterion 'valid_ag-ab_mt_bleu,10' --validation_metrics 'valid_ag-ab_mt_bleu' --amp 0 --master_port 12340
