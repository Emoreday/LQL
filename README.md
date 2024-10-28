# How to quantitize a LLM 

## Introduction

This is a repo for quantitizing a LLM.

## How to use

### Case 1: Quantitize a LLM online

### Case 2: Quantitize a LLM offline

1. Download the model from huggingface
2. Modify the `quantize.py` file, focus on `model_dir` and `quantized_dir`
3. Run `python quantize.py` for quantitization, which will read the model from `model_dir` and save the quantized model to `quantized_dir`
4. Modify the `LLMConfig.yaml`. Focus on `model_loading`-`use_path`-`path`. It should be the path to the quantized model.
5. Run `./LM-eval.py` for evaluation
