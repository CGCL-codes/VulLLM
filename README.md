# VulLLM

This is the codebase for the paper "Generalization-Enhanced Code Vulnerability Detection via Multi-Task Instruction Fine-Tuning".

**Reproduction of Baseline Models**

For the reproduction of the baselines, we refer to their official implementation or [CodeXGLUE](https://github.com/microsoft/CodeXGLUE). Here we take [CodeBERT](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection) as an example and give its fine-tuning and inference scripts.

***Fine-tuning***
```
cd CodeBERT\code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --train_data_file=../../dataset/MixVul/llm/train_512.json \
    --eval_data_file=../../dataset/MixVul/llm/valid_512.json \
    --test_data_file=../../dataset/MixVul/llm/test_512.json \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
```

***Inference***

```
cd CodeBERT\code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_eval \
    --do_test \
    --train_data_file=../../dataset/MixVul/llm/train_512.json \
    --eval_data_file=../../dataset/MixVul/llm/valid_512.json \
    --test_data_file=../../dataset/MixVul/llm/test_512.json \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
```

**Running VulLLM**

For CodeLlama, we refer to its official implementation [llama-recipes](https://github.com/facebookresearch/llama-recipes). The location of the training data is located at `/CodeLlama/configs/datasets.py`. We use the alpaca format data set, which is located in `class alpaca_dataset`.

***Fine-tuning CodeLlama and Llama-2***
```
cd CodeLlama
python finetuning.py \
    --use_peft \
    --model_name codellama/CodeLlama-13b-hf \
    --peft_method lora \
    --batch_size_training 32 \
    --val_batch_size 32 \
    --context_length 512 \
    --quantization \
    --num_epochs 3 \
    --output_dir codellama-13b-multi-r16
```

***Inference***
```
cd CodeLlama
python inference-basic.py \
    --model_type codellama \
    --base_model codellama/CodeLlama-13b-hf \
    --tuned_model odellama-13b-multi-r16 \
    --data_file ../dataset/ReVeal/test_512.json
```

**Adversarial Attacks**

For adversarial attacks, you can run the script `attack_{models}_{attacks}.py` located in the Attack directory, where models can be either `llm` or `ptm`, representing attacks on large language models and pre-trained models, respectively. attacks can be chosen from `mhm`, `wir`, and `deadcode`, which represent two types of attacks based on random identifier replacements and one type of attack based on dead code insertion, respectively. A script example is as follows.
```
cd Attack
python attack_ptm_wir.py \
    --model_type=roberta \
    --output_dir=../CodeBERT/saved_models/ \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --csv_store_path attack_results_ptm/attack_WIR_CodeBERT_ReVeal.csv \
    --eval_data_file=../dataset/ReVeal/test_512.json \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 12345 2>&1 | tee attack_results_ptm/attack_wir_CodeBERT_ReVeal.log
```
The tuned model weights, performance evaluation results, and adversarial attack results are available in [Renodo](https://zenodo.org/records/10677069).

## Acknowledgement

We are very grateful that the authors of CodeLlama, Llama-2, StarCoder make their code publicly available so that we can build our VulLLM on top of their code.