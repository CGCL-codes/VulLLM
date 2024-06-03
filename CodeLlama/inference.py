import torch
import os
import sys
import json
import copy
import pandas as pd
import argparse
from peft import PeftModel
from transformers import GenerationConfig
from transformers import LlamaTokenizer, CodeLlamaTokenizer, LlamaForCausalLM,AutoTokenizer,AutoModelForCausalLM
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#
MODEL_CLASSES = {
    'llama': (LlamaForCausalLM, LlamaTokenizer),
    'codellama': (LlamaForCausalLM, CodeLlamaTokenizer)
}
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
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate code vulnerability detection")
    parser.add_argument("--model_type", default="llama", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--base_model", required=True, help="Path to the base model.")
    parser.add_argument("--tuned_model", required=True, help="Path to the tuned model.")
    parser.add_argument("--data_file", required=True, help="Path to the json file containing test dataset.")
    parser.add_argument("--csv_path", default='results.csv', help="Path to save the CSV results.")
    return parser.parse_args()

def main():
    args = parse_args()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        pad_token_id=tokenizer.eos_token_id
    )
    print("Base model loaded!")

    model = PeftModel.from_pretrained(model, args.tuned_model)
    print("Tuned model loaded!")

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    with open(args.data_file, "r") as file:
        data = json.load(file)

    label_list = []
    prediction_list = []
    print(len(data))

    # Create the CSV header if file doesn't exist
    if not os.path.exists(args.csv_path):
        with open(args.csv_path, 'w') as f:
            f.write('Index,Code,Label,Prediction,Prob,Response\n')

    for i, example_content in enumerate(data):
        prediction_result = 0

        ann = example_content
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)

        model_input = tokenizer(prompt, return_tensors="pt").to(device)

        print("*" * 50, i, "*" * 50, flush=True)
        print("Length: ", model_input['input_ids'].size(1))
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=model_input['input_ids'],
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=100,
            )
            logits = generation_output.scores
            probabilities = [torch.softmax(logit, dim=-1) for logit in logits]
            prob_dist = probabilities[-2]
            confidence_score, predicted_token_id = torch.max(prob_dist, dim=-1)

            s = generation_output.sequences[0]
            generated_response = tokenizer.decode(s)
            response = generated_response.split("### Response:")[-1].strip()
            if len(response) > 0:
                prediction = response[0]
                # 检查prediction是否为'1'或'0'
                if response[0] == '1':
                    prediction_result = 1

            prediction_list.append(prediction_result)
            expected_response = example_content.get("output", "").strip()
            label_list.append(1 if expected_response == '1' else 0)

            print("Label: {} - Response: {}".format(expected_response,prediction_result))

            code = example_content.get("input", "")
            temp_df = pd.DataFrame({'Index': i, 'Code': [code], 'Label': [expected_response], 'Prediction': [prediction_result],
                                    'Prob': [confidence_score.item()], 'Response': [response]})
            temp_df.to_csv(args.csv_path, index=False, mode='a', header=False)

        if i > 0 and i % 100 == 0:
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
    main()
