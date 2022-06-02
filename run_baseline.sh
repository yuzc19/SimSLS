export CUDA_VISIBLE_DEVICES=4

CUDA_LAUNCH_BLOCKING=1 python baseline/main.py --do_eval True

CUDA_LAUNCH_BLOCKING=1 python baseline/main.py