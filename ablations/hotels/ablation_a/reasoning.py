from openai import OpenAI
import time
import json
from tqdm import tqdm
from typing import List


def gpt4_prompting(review_fragments: List, source_documents: List, facet: str):
    prompt_format = open(f"prompts_reasoning/prompt_reasoning_{facet}.txt").read()
    review_text = "\n".join(source_documents)
    prompt_content = prompt_format.replace("{{source_documents}}", review_text)
    # print(prompt_format)
    while True:
        try:
            output_dict = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                messages=[
                    {"role": "system", "content": "Always answer with only the summary, without other content."},
                    {"role": "user",
                     "content": prompt_content}
                    ],
                n=8
                )
            all_candidates = []
            all_candidates_len = []
            tmp = []
            for choice in output_dict.choices:
                output_content = choice.message.content
                content_len = len(output_content.split())
                all_candidates_len.append(content_len)
                tmp.append(content_len)
                all_candidates.append(output_content)
            tmp.sort()
            meta_generated = all_candidates[all_candidates_len.index(tmp[int(len(tmp) / 2)])]
            break
        except Exception as e:
            print(e)
            if ("limit" in str(e)):
                time.sleep(2)
    print(meta_generated)
    return meta_generated


def facet_reasoning(categorization_pairs: List, source_documents: List) -> List:
    result = []
    for pair in categorization_pairs:
        facet = pair["facet"]
        review_fragments = pair["review_fragments"]
        pair["meta_generated"] = gpt4_prompting(review_fragments, source_documents, facet.lower())
        result.append(pair)

    return result


if __name__ == "__main__":
    model_name = "llama31_70b"
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

    with open(f"../../../modular_llama3/hotels/space_selection_result_{model_name}.json") as f:
        test_samples = json.load(f)

    results = []
    for sample in tqdm(test_samples):
        categorization_pairs = sample["categorization_pairs"]
        source_documents = sample["source_documents"]
        sample["categorization_pairs"] = facet_reasoning(categorization_pairs, source_documents)
        results.append(sample)
        # print(sample)

    print(len(results))
    with open(f"space_reasoning_result_{model_name}.json", "w") as f:
        json.dump(results, f, indent=4)
