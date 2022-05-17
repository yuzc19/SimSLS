export CUDA_VISIBLE_DEVICES=0

CUDA_LAUNCH_BLOCKING=1 python inference.py

# ps -up `nvidia-smi -q -x | grep pid | sed -e 's/<pid>//g' -e 's/<\/pid>//g' -e 's/^[[:space:]]*//'`