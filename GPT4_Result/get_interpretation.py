import os
import openai
import re
import pandas as pd


def setup_environment():
    openai.api_key = "your-openai-api-key"
    openai.organization = "your-openai-organization"


def request_result(conversation, prompt_text):
    conversation.append({'role': 'user', "content": prompt_text})
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=conversation,
    )
    conversation.append(
        {"role": "assistant", "content": response['choices'][0]['message']['content']}
    )
    result = response['choices'][0]['message']['content'].replace('\n', ' ').strip()
    return conversation, result


def prompt_1_fun(code):
    prompt = 'Is the following program buggy?\n' + code
    return prompt


def prompt_2_fun():
    prompt = ('This program is buggy. Please double-check the answer and analyze its correctness.'
              'Next, please give the description of vulnerability.')
    return prompt


def prompt_3_fun(cve_description):
    prompt = (f'The description of vulnerability is: {cve_description}'
              'Please double-check the answer and analyze its correctness. '
              'Next, please provide the lines of code that are directly pertinent to the identified vulnerability.')
    return re.sub(r'\n\s*\n', '\n', prompt)


def prompt_4_fun(vulnerability_lines):
    prompt = (f'The vulnerability lines are: \n{vulnerability_lines}'
              'Please double-check the answer and analyze its correctness. '
              'Next, please provide the vulnerability lines related data dependency and control dependency lines.')
    return re.sub(r'\n\s*\n', '\n', prompt)


def prompt_5_fun(dependency_lines):
    prompt = (f'The dependency lines are:\n{dependency_lines} '
              'Please double-check the answer and analyze its correctness. '
              'Next, considering the vulnerability\'s description, please present the vulnerability interpretation by referring to the vulnerable and dependent lines. '
              )
    return prompt


def main():
    setup_environment()
    df = pd.read_csv('vul-data/vul_fun.csv')
    code_list = df['Code'].tolist()
    cve_list = df['CVE'].tolist()
    cve_list_description = df['Description'].tolist()
    vul_lines_list = df['Vulnerability lines'].tolist()
    dep_lines_list = df['Dependency lines'].tolist()
    print(len(code_list), len(cwe_list), len(vul_lines_list), len(dep_lines_list))

    csv_path = 'vul-data/results.csv'
    file_exists = os.path.exists(csv_path)

    for i, code in enumerate(code_list):
        CVE = cve_list[i]
        CVE_description = cve_list_description[i]
        vul_lines = vul_lines_list[i]
        dependency_lines = dep_lines_list[i]

        system_role = dict({'role': 'system', "content": "I want you to act as a vulnerability detection model."})
        conversation = [system_role]

        prompt_1 = prompt_1_fun(code)
        conversation, result_1 = request_result(conversation, prompt_1)
        
        prompt_2 = prompt_2_fun()
        conversation, result_2 = request_result(conversation, prompt_2)
        
        prompt_3 = prompt_3_fun(CVE_description)
        conversation, result_3 = request_result(conversation, prompt_3)
        
        prompt_4 = prompt_4_fun(vul_lines)
        conversation, result_4 = request_result(conversation, prompt_4)
        
        prompt_5 = prompt_5_fun(dependency_lines)
        conversation, result_5 = request_result(conversation, prompt_5)

        result_data = {
            'CVE': CVE,
            'Code': code,
            'Result': result_5
        }

        temp_df = pd.DataFrame([result_data])
        
        if not file_exists:
            temp_df.to_csv(csv_path, index=False, mode='w')
            file_exists = True
        else:
            temp_df.to_csv(csv_path, index=False, mode='a', header=False)

if __name__ == "__main__":
    main()
