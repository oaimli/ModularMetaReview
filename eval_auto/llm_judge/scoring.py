import json
from elo_rating import Elo


def scoring(samples):
    # The calculation is based on the library of elo-rating
    e = Elo()
    result_models = {}
    for sample in samples:
        comparisons = sample["comparisons"]
        for comparison in comparisons:
            model_a = comparison["model_a"]
            model_b = comparison["model_b"]
            better_one = comparison["better"]

            tmp = result_models.get(model_a, [])
            if better_one == model_a:
                tmp.append(1)
                e.add_match(model_a, model_b, 1.0, k=0.15)
            else:
                tmp.append(0)
                e.add_match(model_a, model_b, 0.0, k=0.15)
            result_models[model_a] = tmp

            tmp = result_models.get(model_b, [])
            if better_one == model_b:
                tmp.append(1)
            else:
                tmp.append(0)
            result_models[model_b] = tmp

    winning_rates = {}
    for model, result in result_models.items():
        winning_rates[model] = sum(result) / len(result)

    winning_rates = sorted(winning_rates.items(), key=lambda x: x[1], reverse=True)

    elo_rankings = e.rankings()

    return winning_rates, elo_rankings


if __name__ == "__main__":

    with open("../info.json") as f:
        info = json.load(f)

    dataset_names = ["space", "peermeta", "amasum_shoes"]
    for dataset_name in dataset_names:
        print(dataset_name)
        output_name = f"{dataset_name}_llm_judged.json"
        print(output_name)
        with open(output_name) as f:
            all_samples = json.load(f)
        winning_rates, elo_rankings = scoring(all_samples)
        print("winning rates", winning_rates)
        print("elo rankings", elo_rankings)
