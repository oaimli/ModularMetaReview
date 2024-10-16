import os.path
import jsonlines
from openai import OpenAI
import time
import json
from tqdm import tqdm
from typing import List


def meta_generation(source_documents: List) -> str:
    document_groups = []
    document_count = len(source_documents)
    group_size = int(document_count / 5)
    for i in range(5):
        document_group = source_documents[i * group_size: (i + 1) * group_size]
        document_groups.append(document_group)

    small_summaries = []
    for document_group in document_groups:
        source_text = "\n".join(document_group)
        prompt_content = f"Please write a summary for the following reviews on sports shoes.\n\n Reviews on a pair of sports shoes:\n {source_text}\n\nThe output summary:"
        # print(prompt_format)
        while True:
            try:
                output_dict = client.chat.completions.create(
                    model="gpt-4o-2024-05-13",
                    messages=[
                        {"role": "system",
                         "content": "You are requested to do summarization. Please output the final answer with only the summary, no other useless content."},
                        {"role": "user",
                         "content": prompt_content}
                        ],
                    n=1
                    )
                output = output_dict.choices[0].message.content
                small_summaries.append(output)
                break
            except Exception as e:
                if "limit" in str(e):
                    time.sleep(2)

    source_text = "\n".join(small_summaries)
    prompt_content = f"Please write a summary for the following texts.\n\n The texts:\n {source_text}\n\nThe output summary:"
    # print(prompt_format)
    while True:
        try:
            output_dict = client.chat.completions.create(
                model="gpt-4o-2024-05-13",
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
    model_name = "gpt_4o"
    client = OpenAI(api_key="sk-proj-jxdkj7TzTCWDjDU0lpEPT3BlbkFJll01Dz3fxt51wM8Rh6wm")

    test_samples = []
    with jsonlines.open("../../datasets/amasum_shoes_test.jsonl") as reader:
        for line in reader:
            test_samples.append(line)

    results = []
    for sample in tqdm(test_samples):
        source_documents = sample["source_documents"]
        sample["generated_meta_review"] = meta_generation(source_documents)
        results.append(sample)
        # print(sample)

    print(len(results))
    output_dir = "../../results/llama3_pr_amasum_shoes"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f"{output_dir}/generations_{model_name}_chunk.json", "w") as f:
        json.dump(results, f, indent=4)
