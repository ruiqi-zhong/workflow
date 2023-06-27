#!/bin/sh
eval "$(conda shell.bash hook)"
#conda activate /scratch/users/ruiqi-zhong/conda/envs/qlora
conda activate /scratch/users/ruiqi-zhong/mini/envs/1223seq2seq/
set -x

# You don't need to change the lines above

python3 seq2seq.py --model_init_path t5-small --data_name data --eval_steps 500 --save_steps 1000 --max_steps 10001
python3 seq2seq.py --model_init_path t5-large --data_name data --eval_steps 500 --save_steps 1000 --max_steps 10001
