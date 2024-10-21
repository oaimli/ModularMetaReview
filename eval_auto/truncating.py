import numpy as np
import json


if __name__ == "__main__":
    with open("info.json") as f:
        info = json.load(f)

    dataset_names = ["peermeta", "space", "amasum_shoes"]
    for dataset_name in dataset_names:
        print(dataset_name)
        avg_length_reference = -1
        generations_info = info[dataset_name]
        for generation_info in generations_info:
            generation_file = generation_info["generation_file"]
            print(generation_file)
            candidate_key = generation_info["candidate_key"]
            reference_key = generation_info["reference_key"]
            with open(generation_file) as f:
                samples = json.load(f)

            if avg_length_reference == -1:
                lengths_reference = []
                for i, sample in enumerate(samples):
                    # calculate the length of the reference
                    reference = ""
                    if isinstance(sample[reference_key], list):
                        reference = sample[reference_key][0] # SPACE has multiple references
                    else:
                        reference = sample[reference_key]
                    lengths_reference.append(len(reference.split()))
                avg_length_reference = np.mean(lengths_reference)

            lengths_diff = []
            for i, sample in enumerate(samples):
                candidate = sample[candidate_key]
                sample[candidate_key] = " ".join(candidate.split()[:int(avg_length_reference)])
                samples[i] = sample

                diff = len(candidate.split()) - avg_length_reference
                if diff > 0:
                    lengths_diff.append(diff)
            if len(lengths_diff) > 0:
                print(np.mean(lengths_diff))
            else:
                print(0)
            # save truncated results
            output_file = generation_file[:-5] + "_truncated.json"
            with open(output_file, "w") as f:
                json.dump(samples, f, indent=4)