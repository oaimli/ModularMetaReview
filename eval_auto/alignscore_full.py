import numpy as np
import json
from alignscore import AlignScore


if __name__ == "__main__":
    scorer = AlignScore(model='roberta-large', batch_size=32, device='cuda:0', ckpt_path='../tmp/AlignScore/AlignScore-large.ckpt',
                        evaluation_mode='nli_sp')

    with open("info.json") as f:
        info = json.load(f)

    dataset_names = ["space", "peermeta", "amasum_shoes"]
    for dataset_name in dataset_names:
        scores_source = {}
        print(dataset_name)

        generations_info = info[dataset_name]

        # human reference
        generation_file = generations_info[0]["generation_file"]
        print("human reference")
        reference_key = generations_info[0]["reference_key"]
        source_key = "source_documents"
        with open(generation_file) as f:
            samples = json.load(f)
        references = []
        source_texts = []
        for sample in samples:
            if isinstance(sample[reference_key], str):
                references.append(sample[reference_key])
            else:
                references.append(sample[reference_key][0])  # SPACE has multiple references
            source_texts.append("\n".join(sample[source_key]))
        # compared with source texts
        scores_align = scorer.score(contexts=source_texts, claims=references)
        print(scores_align)
        scores_source["human_reference"] = scores_align
        print("scores alignscore:", np.mean(scores_align))

        # model generations
        for generation_info in generations_info:
            generation_file = generation_info["generation_file"]
            print(generation_file)
            with open(generation_file) as f:
                samples = json.load(f)
            candidate_key = generation_info["candidate_key"]
            source_key = "source_documents"

            candidates = []
            source_texts = []
            for sample in samples:
                candidates.append(sample[candidate_key])
                source_texts.append("\n".join(sample[source_key]))

            scores_align = scorer.score(contexts=source_texts, claims=candidates)
            scores_source[generation_file] = scores_align
            print("scores alignscore", np.mean(scores_align))


        with open(f"{dataset_name}_alignscore_full.json", "w") as f:
            json.dump(scores_source, f, indent=4)
