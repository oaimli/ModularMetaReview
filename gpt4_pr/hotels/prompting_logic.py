import jsonlines
from openai import OpenAI
import time
import json
from tqdm import tqdm
from typing import List
import jsonlines


def meta_generation(source_documents: List) -> str:
    prompt_format = open("prompt_logic.txt").read()
    source_text = "\n".join(source_documents)
    prompt_content = prompt_format.replace("{{source_documents}}", source_text)
    # print(prompt_format)
    while True:
        try:
            output_dict = client.chat.completions.create(
                model="gpt-4o",
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
    model_name = "gpt4"
    client = OpenAI(api_key="sk-proj-jxdkj7TzTCWDjDU0lpEPT3BlbkFJll01Dz3fxt51wM8Rh6wm")

    test_samples = []
    with jsonlines.open("../../datasets/space_test.jsonl") as reader:
        for line in reader:
            test_samples.append(line)

    results = []
    for sample in tqdm(test_samples):
        source_documents = sample["source_documents"]
        sample["generated_summary_general"] = meta_generation(source_documents)
        results.append(sample)
        # print(sample)

    print(len(results))
    with open(f"gpt4_pr_space/generations_{model_name}_logic.json", "w") as f:
        json.dump(results, f, indent=4)
