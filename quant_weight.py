import os
import gc
import torch
from quant import block_minifloat_quantize, BlockMinifloat

# 设置环境变量
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:27890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:27890'
os.environ['HF_HOME'] = '/home/dataset/huggingface'
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

# BFP 参数设置
bfp_setup = True
EBIT = 4
MBIT = 3
BLOCK_SIZE = 8
BFP_number = BlockMinifloat(EBIT, MBIT, BLOCK_SIZE)
ems_setup = True

# 配置字典
config = {
    "bfp": bfp_setup,
    "ehbfp": ems_setup,
    "EBIT": EBIT,
    "MBIT": MBIT,
    "BLOCK_SIZE": BLOCK_SIZE,
}

# 模型参数文件所在目录 （Oringal_PATH)
model_dir = '/home/dataset/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/'
# 量化后的文件保存目录 （Quant_PATH)
quantized_dir = '/home/dataset/huggingface/quant/models--meta-llama--Llama-3.1-8B-hebfp/snapshots/'
os.makedirs(quantized_dir, exist_ok=True)

# 模拟的量化函数
def quantize_tensor(weights):
    return block_minifloat_quantize(weights, BFP_number, ems=ems_setup).to(torch.float16)

# 筛选所有 .bin 和 .safetensors 文件
files = [f for f in os.listdir(model_dir) if f.endswith(('.bin', '.safetensors')) and 'index' not in f]

# 遍历并量化每个文件
for file in files:
    file_path = os.path.join(model_dir, file)
    print(f'Loading {file}...')

    # 加载模型参数
    if file.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(file_path)  # 使用 safetensors 加载
    else:
        state_dict = torch.load(file_path, map_location='cpu')  # 处理 .bin 文件

    # 将字典分割为两部分
    keys = list(state_dict.keys())
    mid_index = len(keys) // 2
    first_half_keys = keys[:mid_index]
    second_half_keys = keys[mid_index:]

    # 量化并处理每个键
    print(f'Quantizing {file}...')
    quantized_state_dict = {}

    for key, tensor in state_dict.items(): 
        print(f"Key is {key}")
        if isinstance(tensor, torch.Tensor):
            # 检查是否是要排除的部分
            if key == 'model.embed_tokens.weight' or key == 'lm_head.weight':
                quantized_state_dict[key] = tensor.to('cpu')  # 不进行量化，直接存储
            else:
                device = 'cuda:0' if key in first_half_keys else 'cuda:1'
                tensor = quantize_tensor(tensor.to(device))
                quantized_state_dict[key] = tensor.to('cpu')
        else:
            quantized_state_dict[key] = tensor

    quantized_state_dict = {k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in quantized_state_dict.items()}

    # 保存量化后的权重
    if file.endswith('.safetensors'):
        from safetensors.torch import save_file
        quantized_path = os.path.join(quantized_dir, file)  # 保持 Safetensors 格式
        save_file(quantized_state_dict, quantized_path, {"format": "pt"})
    else:
        quantized_path = os.path.join(quantized_dir, file)  # 保持 .bin 格式
        torch.save(quantized_state_dict, quantized_path)

    # 释放内存
    del state_dict
    del quantized_state_dict
    gc.collect()
    torch.cuda.empty_cache()

    print(f'{file} quantization completed.')

print('All files have been quantized.')
