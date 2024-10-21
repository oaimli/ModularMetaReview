import numpy as np
import json


if __name__ == "__main__":
    with open("info.json") as f:
        info = json.load(f)

    dataset_names = ["peermeta", "space", "amasum_shoes"]
    for dataset_name in dataset_names:
        print(dataset_name)
        generations_info = info[dataset_name]
        for generation_info in generations_info:
            generation_file = generation_info["generation_file"]
            print(generation_file)
            candidate_key = generation_info["candidate_key"]
            reference_key = generation_info["reference_key"]
            with open(generation_file) as f:
                samples = json.load(f)
            lengths_diff = []
            for i, sample in enumerate(samples):
                # calculate the length of the reference
                references = []
                if isinstance(sample[reference_key], list):
                    references.extend(sample[reference_key]) # SPACE has multiple references
                else:
                    references.append(sample[reference_key])
                lengths_reference = []
                for reference in references:
                    lengths_reference.append(len(reference.split()))
                avg_length_reference = int(np.mean(lengths_reference))

                candidate = sample[candidate_key]
                sample[candidate_key] = " ".join(candidate.split()[:avg_length_reference])
                samples[i] = sample

                lengths_diff.append(abs(avg_length_reference - len(candidate.split())))
            print(np.mean(lengths_diff))
            # save truncated results
            output_file = generation_file[:-5] + "_truncated.json"
            with open(output_file, "w") as f:
                json.dump(samples, f, indent=4)