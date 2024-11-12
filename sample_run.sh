modelname="resnet34"
# change sb eb to include more samples in sensitivity set (#samples = (eb-sb+1)*bs)
# we use 1 batch (64 samples) as a quick sample run
# please use 1024 samples
sb=0 # if more than 1 batches
eb=0 # multi gpu parrelization is encouraged
dp="/home/usr1/xxxx/data/imagenet" # example of imagenet datapth, change this to the local folder
echo "sensitivity preparation : MPQCO and CLADO"
python3 prep_mpqco_clado.py  --datapath $dp --modelname $modelname --cuda 0 --start-batch $sb --end-batch $eb --bs 64 --nthreads 8 &> mpqco_clado_prep.log
echo "sensitivity preparation : HAWQV2/3"
python3 prep_hawq.py --datapath $dp --modelname $modelname --start-batch $sb --end-batch $eb --bs 64 --nthreads 8 --cuda 0 &> hawq_prep.log
echo "IQP(CLADO)/ILP(HAWQ,MPQCO) Optimization and Evaluation on Test..."
python3 optimize.py --datapath $dp --modelname $modelname --bs 64 --nthreads 8 --start-batch $sb --end-batch $eb --cuda 0 &> optimize.log
