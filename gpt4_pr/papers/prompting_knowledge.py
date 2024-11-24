import os.path
from openai import OpenAI
import time
import json
from tqdm import tqdm
from typing import List
import jsonlines


def meta_generation(source_documents: List) -> str:
    prompt_format = open("prompt_knowledge.txt").read()
    source_text = "\n".join(source_documents)
    prompt_content = prompt_format.replace("{{source_documents}}", source_text)
    # print(prompt_format)
    while True:
        try:
            output_dict = client.chat.completions.create(
                model="gpt-4o-2024-05-13",
                messages=[
                    {"role": "system",
                     "content": "You are requested to write the summary. Please output the final answer with only the summary, no other useless content."},
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
    with jsonlines.open("../../datasets/peermeta_test.jsonl") as reader:
        for line in reader:
            test_samples.append(line)

    results = []
    for sample in tqdm(test_samples):
        source_documents = sample["source_documents"]
        sample["generated_meta_review"] = meta_generation(source_documents)
        results.append(sample)
        # print(sample)

    print(len(results))
    output_dir = "../../results/gpt4_pr_peermeta"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f"{output_dir}/generations_{model_name}_knowledge.json", "w") as f:
        json.dump(results, f, indent=4)
