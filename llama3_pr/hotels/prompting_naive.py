import os.path
import jsonlines
from openai import OpenAI
import time
import json
from tqdm import tqdm
from typing import List


def meta_generation(source_documents: List, aspect: str) -> str:
    source_text = "\n".join(source_documents)
    if aspect == "general":
        prompt_content = f"Please write a summary for the following reviews on a hotel.\n\n Reviews on a hotel:\n {source_text}\n\nThe output summary:"
        # print(prompt_format)
        while True:
            try:
                output_dict = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                    messages=[
                        {"role": "system",
                         "content": "You are requested to do summarization. Please output the final answer with only the summary, no other useless content."},
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
    else:
        prompt_content = f"Please write a summary for the following reviews on a hotel for the aspect of {aspect}.\n\n Reviews on a hotel:\n {source_text}\n\nThe output summary:"
        # print(prompt_format)
        while True:
            try:
                output_dict = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                    messages=[
                        {"role": "system",
                         "content": "You are requested to do summarization. Please output the final answer with only the summary, no other useless content."},
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
    print(output)
    return output


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
        sample["generated_summary_general"] = meta_generation(source_documents, "general")
        sample["generated_summary_service"] = meta_generation(source_documents, "service")
        sample["generated_summary_rooms"] = meta_generation(source_documents, "rooms")
        sample["generated_summary_location"] = meta_generation(source_documents, "location")
        sample["generated_summary_food"] = meta_generation(source_documents, "food")
        sample["generated_summary_cleanliness"] = meta_generation(source_documents, "cleanliness")
        sample["generated_summary_building"] = meta_generation(source_documents, "building")
        results.append(sample)
        # print(sample)

    print(len(results))
    output_dir = "../../results/llama3_pr_space"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f"{output_dir}/generations_{model_name}_naive.json", "w") as f:
        json.dump(results, f, indent=4)
