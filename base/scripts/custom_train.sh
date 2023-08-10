export CUDA_VISIBLE_DEVICES=0,1,2,3

# kill -9 $(lsof -t /dev/nvidia*)
# sleep 1s
# kill -9 $(lsof -t /dev/nvidia*)
# sleep 1s

config=configs/swin_small_jointtraining.py

python3 -m paddle.distributed.launch --log_dir=./logs/swin_small_jointtraining --gpus="0,1,2,3"  tools/ufo_train.py --config-file ${config} --resume 


