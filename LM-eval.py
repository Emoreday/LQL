#!/usr/bin/env python
# coding: utf-8

# In[3]:


# select mult GPUS
import os
import yaml
from accelerate.utils import offload_state_dict

# 读取 YAML 配置文件
with open('LLMConfig.yaml', 'r') as file:
    config = yaml.safe_load(file)
print(f"[INFO] Loading Config Successfully.")

# 设置环境变量
os.environ['HTTP_PROXY'] = config['os']['http_proxy']
os.environ['HTTPS_PROXY'] = config['os']['https_proxy']
os.environ['HF_HOME'] = config['os']['hf_home']
os.environ['CUDA_VISIBLE_DEVICES'] = config['os']['cuda_visible_devices']
os.environ['TRANSFORMERS_CACHE'] = config['os']['transformers_cache']

from quant import block_minifloat_quantize, BlockMinifloat
import torch
from huggingface_hub import login

# 设置 BFP 参数
bfp_setup = config['bfp']['enabled']
EBIT = config['bfp']['ebit']
MBIT = config['bfp']['mbit']
BLOCK_SIZE = config['bfp']['block_size']
ems_setup = config['ehb']['enabled']
single = config['bfp']['single']
BFP_number = BlockMinifloat(EBIT, MBIT, BLOCK_SIZE)  # init BFP

# 设置模型读取参数
if config['model_loading']['use_name']['enable']:
    model_full_name = config['model_loading']['use_name']['name']
    model_name = model_full_name.split('/')[1]
    co_name = model_full_name.split('/')[0]
    activation_quant = config['model_loading']['use_name']['activation_quant']
elif config['model_loading']['use_path']['enable']:
    model_path = config['model_loading']['use_path']['path']
    try:
        if model_path.endswith('/'):
            co_name = model_path.split('/')[-3].split('--')[1]
            model_name = model_path.split('/')[-3].split('--')[2]
        else:
            co_name = model_path.split('/')[-2].split('--')[1]
            model_name = model_path.split('/')[-2].split('--')[2]
    except IndexError:
        print("[ERROR] No name found in the path. Please check your path.")
        exit()
    offload_folder = config['model_loading']['use_path']['offload_folder']
    activation_quant = config['model_loading']['use_path']['activation_quant']
else:
    raise ValueError("[ERROR] Config ERROR on model_loading: no Available method to load model. ")

access_token = config['huggingface']['token']

# In[]

file_name = f"results/{co_name}/{model_name}/results-e{EBIT}m{MBIT}-{single}"
if os.path.exists(f"{file_name}.json"):
    raise FileExistsError(f"文件已存在: {file_name}")
os.makedirs(os.path.dirname(f'{file_name}.json'), exist_ok=True)
login(token=access_token)


# In[ ]:

# init BFP Activation function
def custom_hook(module, input, output):
    return block_minifloat_quantize(output, BFP_number, ems=False)

from lm_eval.models.huggingface import HFLM
class CustomModel(HFLM):
    def __init__(self, model_name, dtype="auto", batch_size=4, max_batch_size=64,
                 parallelize=False, max_memory_per_gpu=None,
                 weight_quant=False, activation_quant=False):
        super().__init__(model_name, dtype=dtype, batch_size=batch_size, max_batch_size=max_batch_size,
                         parallelize=parallelize, max_memory_per_gpu=max_memory_per_gpu)
        # 量化权重
        if weight_quant:
            for name, module in self.model.named_modules():
                if isinstance(module, (torch.nn.LayerNorm, torch.nn.Linear)):
                    # 量化权重
                    module.weight.data = block_minifloat_quantize(module.weight.data, BFP_number, ems=ems_setup)
        # 量化 Activation
        if activation_quant:
            for name, module in self.model.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
                    # 检查是否是最后一个线性层
                    # if not (isinstance(module, torch.nn.Linear) and name == 'lm_head'):
                    module.register_forward_hook(custom_hook)
        print("Successful registration of hook!")
        self.model.half()


# In[ ]:
import lm_eval

# Using model_name
if config['model_loading']['use_name']['enable']:
    print(f'[INFO] Using model name: {model_name}')
    if not bfp_setup:
        lm_obj = lm_eval.models.huggingface.HFLM(pretrained=model_full_name,
                                                 device='cuda',
                                                 dtype=torch.float16,
                                                 max_batch_size=64,
                                                 parallelize=True
                                                 )
    if bfp_setup:
        lm_obj = CustomModel(model_full_name,
                             dtype=torch.float16,
                             max_batch_size=64,
                             parallelize=True,
                             max_memory_per_gpu=None,
                             weight_quant=bfp_setup,
                             activation_quant=activation_quant)
elif config['model_loading']['use_path']['enable']:
    print(f'[INFO] Using model path: {model_path}')
    from transformers import AutoModelForCausalLM
    # max_memory = {i: "24GB" for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        offload_folder=offload_folder,
        local_files_only=True,
        # max_memory=max_memory
    )
    lm_obj = CustomModel(model, dtype=torch.float16, batch_size=4, max_batch_size=64)

print(f'[INFO] Success Loading Model: {model_name}')


# In[ ]:
# TaskManager 用于管理任务daan
task_manager = lm_eval.tasks.TaskManager()

# 评估模型
# 需要在 tasks 内部指定对应任务
results = lm_eval.simple_evaluate(
    model=lm_obj,
    tasks=tasks,
    num_fewshot=0,
    task_manager=task_manager,
    limit=100,
    batch_size=128
)


# In[ ]:
import json

del results['samples'], results['configs']
results['config']['model_dtype'] = str(results['config']['model_dtype'])
results['method Config'] = config

json_str = json.dumps(results, indent=4)

# 将字典对象保存为JSON文件
with open(f'{file_name}.json', 'w') as json_file:
    json_file.write(json_str)

print(f"Results have been saved to {file_name}.json.")
