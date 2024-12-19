"""
Implementation of the decomposed prompting work
"""
import os.path
import jsonlines
from openai import OpenAI
import time
import json
from tqdm import tqdm
from typing import List


def meta_generation(source_documents: List, aspect: str) -> (str, str):
    # aspect is not taken into consideration here
    source_text = "\n".join(source_documents)
    prompt_content = f"Please give me sequential steps to write a summary specific for the following reviews on a hotel.\n\n Reviews on a hotel:\n {source_text}\n\nThe steps to write a summary in different lines:"
    # print(prompt_format)
    while True:
        try:
            output_dict = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                messages=[
                    {"role": "system",
                     "content": "You are requested to write the steps. Please output the final answer with only the steps in different lines, no other useless content."},
                    {"role": "user",
                     "content": prompt_content}
                    ],
                n=1
                )
            steps = output_dict.choices[0].message.content
            break
        except Exception as e:
            if "limit" in str(e):
                time.sleep(2)
    print(steps)

    steps_list = []
    for step in steps.split("\n"):
        if step.strip() != "":
            steps_list.append(step)

    for step_id, step in enumerate(steps_list):
        if step_id == 0:
            prompt_content = f"{source_text}\nPlease follow the instruction below and give your output.\n {step}\nThe output:"
        else:
            prompt_content = f"{output}\nPlease follow the instruction below and give your output.\n {step}\nThe output:"
        # print(prompt_format)
        while True:
            try:
                output_dict = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                    messages=[
                        {"role": "system",
                         "content": "You are requested to follow the instruction and only generate the requested output."},
                        {"role": "user",
                         "content": prompt_content}
                        ],
                    n=1
                    )
                output = output_dict.choices[0].message.content
                break
            except Exception as e:
                if "limit" in str(e):
                    time.sleep(2)
        print("******************************\n", output)

    print("###############################\n", output)
    return output, steps


if __name__ == "__main__":
    model_name = "llama31_70b"
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        )

    test_samples = []
    with jsonlines.open("../../datasets/space_test.jsonl") as reader:
        for line in reader:
            test_samples.append(line)

    results = []
    for sample in tqdm(test_samples):
        source_documents = sample["source_documents"]
        result, steps = meta_generation(source_documents, "general")
        sample["generated_summary_general"] = result
        sample["generated_steps_general"] = steps
        results.append(sample)
        # print(sample)

    print(len(results))
    output_dir = "../../results/llama3_pr_space"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f"{output_dir}/generations_{model_name}_decomposed.json", "w") as f:
        json.dump(results, f, indent=4)
