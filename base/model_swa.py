import paddle
import copy
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model weight averaging')
    parser.add_argument('--model-dir',  help='the dir to save logs and models')
    args = parser.parse_args()
    return args

def average_model_weights(model_weights_list):
    num_models = len(model_weights_list)
    if num_models == 0:
        raise ValueError("No model weights provided for averaging.")
    
    averaged_state_dict1 = copy.deepcopy(model_weights_list[0])
    averaged_state_dict2 = model_weights_list[0].copy()

    params_ori = averaged_state_dict2['model']
    for idx in range(1, num_models):
        param = model_weights_list[idx]['model']
        for key in params_ori.keys():
            params_ori[key] += param[key]
    for key in params_ori.keys():
        params_ori[key] = params_ori[key] / num_models
    averaged_state_dict2['model'] = params_ori

    for key in averaged_state_dict2['model'].keys():
        averaged_state_dict2['model'][key] = averaged_state_dict2['model'][key].astype(averaged_state_dict1['model'][key].dtype)
    
    return averaged_state_dict2


def main():
    args = parse_args()

    weights_dicts = []
    dir = args.model-dir

    # 取12个epoch的模型求权重平均
    for i in range(12):
        model_name = 'model_0' + str(190749 + 1750 * i) + '.pdmodel'
        model_path = os.path.join(dir, model_name)
        weights_dicts.append(paddle.load(model_path))

    # 将多个模型的权重进行平均
    averaged_state_dict = average_model_weights(weights_dicts)

    # 保存平均后的模型权重到文件
    save_path = "joint_train_averaged_model_weights.pdparams"
    paddle.save(averaged_state_dict, save_path)

if __name__ == "__main__":
    main()

