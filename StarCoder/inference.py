import sys
import os
import fire
import torch
import transformers
import json
import pandas as pd
import jsonlines
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def evaluate(
        batch_data,
        tokenizer,
        model,
        input=None,
        # temperature=1,
        # top_p=0.9,
        # top_k=40,
        num_beams=1,
        max_new_tokens=2048,
        **kwargs,
):
    ann = batch_data
    if ann.get("input", "") == "":
        prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
    else:
        prompt = PROMPT_DICT["prompt_input"].format_map(ann)

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        # temperature=temperature,
        # top_p=top_p,
        # top_k=top_k,
        num_beams=num_beams,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences
    output = tokenizer.batch_decode(s, skip_special_tokens=True)
    return output

def main(
    load_8bit: bool = False,
    base_model: str = "Model_Path",
    input_data_path = "Input.jsonl",
    csv_path = "Output.jsonl",
    tuned_model: str = "tuned model",
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='bigcode/starcoder'"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    if not load_8bit:
        model.half()

    model = PeftModel.from_pretrained(model, tuned_model)
    model.to(device)
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # input_data = jsonlines.open(input_data_path, mode='r')
    with open(input_data_path, "r") as file:
        input_data = json.load(file)

    label_list = []
    prediction_list = []
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write('Code,Label,Prediction,Response\n')
    for num, line in enumerate(input_data):
        print("*" * 50, num, "*" * 50, flush=True)
        prediction_result = 0
        one_data = line
        _output = evaluate(one_data, tokenizer, model)
        final_output = _output[0].split("### Response:")[1].strip()
        if len(final_output) > 0:
            prediction = final_output[0]
            # 检查prediction是否为'1'或'0'
            if final_output[0] == '1':
                prediction_result = 1

        prediction_list.append(prediction_result)
        expected_response = one_data.get("output", "").strip()
        label_list.append(1 if expected_response == '1' else 0)

        print("response:", prediction)
        print("label:", expected_response)

        code = one_data.get("input", "")
        temp_df = pd.DataFrame({'Code': [code], 'Label': [expected_response], 'Prediction': [prediction_result],
                                'Response': [final_output]})
        temp_df.to_csv(csv_path, index=False, mode='a', header=False)
        if num > 0 and num % 100 == 0:
            # Calculate
            print_metrics(label_list, prediction_list)
# Final metrics calculation
    print_metrics(label_list, prediction_list)

def print_metrics(label_list, prediction_list):
    accuracy = accuracy_score(label_list, prediction_list)
    print(f'accuracy：{accuracy:.4f}')

    precision = precision_score(label_list, prediction_list)
    print(f'precision：{precision:.4f}')

    recall = recall_score(label_list, prediction_list)
    print(f'recall：{recall:.4f}')

    f1 = f1_score(label_list, prediction_list)
    print(f'F1：{f1:.4f}')
if __name__ == "__main__":
    fire.Fire(main)