#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --gres=gpu:1
#SBATCH --mem=30g
#SBATCH -t 0

set -x  # echo commands to stdout
set -e  # exit on error

#module load cudnn-8.0-5.1
export CUDA_HOME="/projects/tir1/cuda-8.0.27.1"
PATH=$HOME/bin:$PATH
PATH=$CUDA_HOME/bin:$PATH

export PATH  
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/projects/tir1/cuda/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/yihengz1/anaconda2/pkgs/glibc-2.16.0/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/gcc/5.4.0/lib64:$LD_LIBRARY_PATH


export LD_LIBRARY_PATH=${CUDA_HOME}/include:$LD_LIBRARY_PATH
export CPATH=${CUDA_HOME}/include:$CPATH
export LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
#export CUDA_HOME="/projects/tir1/cuda-8.0.27.1"
#export LD_LIBRARY_PATH=/home/xinyiw1/software/dynet-base/dynet/build/dynet/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/cudnn-8.0/lib64:$LD_LIBRARY_PATH
export CPATH=/opt/cudnn-8.0/include:$CPATH
#export LIBRARY_PATH=/opt/cudnn-8.0/lib64:$LD_LIBRARY_PATH


echo $LD_LIBRARY_PATH
#model_name=base_big
mkdir -p mappings/lf2lf;
#mkdir -p cache/lf2lf;
mkdir -p checkpoint/lf2lf;
PYTHONPATH=. python main.py --schema-path data/craigslist-schema.json --train-examples-paths data/train-parsed.json --test-examples-paths data/dev-parsed.json \
--price-tracker price_tracker.pkl \
--model lf2lf \
--model-path checkpoint/lf2lf --mappings mappings/lf2lf \
--word-vec-size 300 --pretrained-wordvec '' '' \
--rnn-size 300 --rnn-type LSTM --global-attention multibank_general \
--num-context 2 --stateful \
--batch-size 128 --gpuid 0 --optim adagrad --learning-rate 0.01 \
--epochs 15 --report-every 500 \
--cache cache/lf2lf \
--verbose
