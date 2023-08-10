export CUDA_VISIBLE_DEVICES=0,1

# kill -9 $(lsof -t /dev/nvidia*)
# sleep 1s
# kill -9 $(lsof -t /dev/nvidia*)
# sleep 1s

config=configs/test_swin_small_230730_1.py

python3 -m paddle.distributed.launch --log_dir=./logs/vitbase_jointraining --gpus="0,1"  tools/ufo_test.py --config-file ${config} --eval-only 


