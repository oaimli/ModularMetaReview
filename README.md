# Decomposed Opinion Summarization with Verified Aspect-Aware Modules

In this project, we proposed a modular approach to guided by review aspects (e.g., cleanliness for hotel reviews) to achieve more expalinable and grounded opinion summarization in multiple domains.
Our approach separates the tasks of aspect identification, opinion consolidation, and meta-review synthesis to enable greater transparency and ease of inspection, as shown in the figure below.
<img width="967" height="491" alt="image" src="https://github.com/user-attachments/assets/26b54652-c2a9-4e0f-981e-eac5937bec4a" />

This repo contains all the code to implement our approach, other baselines, human annotation and result analysis.
```
/
├── ablations/                 --> (Ablation experiments with Llama3.1-70B-Instruct: detailed explanation in the code)
├── annotations/               --> (Annotated data for intermediate results of meta-reviews, obtained from https://aclanthology.org/2024.acl-long.547/)
├── competitors/               --> (Generation results from previous papers, e.g., HIRO)
├── datasets/                  --> (The preprocessed data is saved here, you may need to download them separately)
├── eval_auto/                 --> (Evaluation results with automatic evaluation metrics)
├── eval_human/                --> (Human evaluation results for the generated meta-reviews)
├── generation_analysis/       --> (Analysis of the generations, e.g., aspect distritbution)   
├── gpt4_pr/                   --> (Baseline approaches based on prompting GPT-4)
├── led_ft/                    --> (Baseline approaches based on finetuning LED)
├── llama3_8b_pr/              --> (Baseline approaches based on prompting Llama3.1-8B-Instruct)
├── llama3_ft/                 --> (Baseline approaches based on finetuning Llama3.1-8B)
├── llama3_pr/                 --> (Baseline approaches based on prompting Llama3.1-70B-Instruct)
├── modular_gpt4/              --> (Our modular approach implemented based on GPT-4 for datasets in the three domains)
├── modular_llama3/            --> (Our modular approach implemented based on Llama3.1-70B-Instruct for datasets in the three domains)
├── modular_llama3_8b/         --> (Our modular approach implemented based on Llama3.1-8B-Instruct for datasets in the three domains)
├── optimization/              --> (A little optimization of the prompts that we used for the modular approach)
├── plots/                     --> (Figures used in the publication)
├── preparation/               --> (Prepare the datasets for the three domains, including PeerSum, Space, and AmaSum)
├── results/                   --> (Results for all the approaches)
├── tcg/                       --> (Results for TCG, a model from Prompted Opinion Summarization with GPT-3.5)
├── utils/                     --> (Scripts for evaluation of generation results)
└── README.md                  --> (This readme file)
```
You can easily get implementation codes for all the approaches with their requirements in specific approach folders. We combine the generation results of different approaches and put them in eval_auto and eval_human. 
Our present results are obtained from the following files:

/eval_auto/peermeta_generations_full.json
/eval_auto/space_generations_full.json
/eval_auto/amasum_shoes_generations_full.json
/eval_human/meta_reviews/generations_amasum_shoes.json
/eval_human/meta_reviews/generations_peermeta.json
/eval_human/meta_reviews/generations_space.json
/eval_human/intermediate/sampled.json
