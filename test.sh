case $SLURM_JOB_PARTITION in
a40)
b=9000
;;
rtx6000)
b=3000
;;
t4v2)
b=2000
;;
interactive)
echo don\'t know
b=2000
;;
esac
echo $b