from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None, metadata={"help": "Name or path of the pre-trained model."})
    max_length_model: Optional[int] = field(default=1536, metadata={"help": "Max input length of the model."})
    # predict with sampling or contrastive search
    max_predict_length: Optional[int] = field(default=128, metadata={
        "help": "Max predicted target length when generation, excluding the source part."})
    min_predict_length: Optional[int] = field(default=1, metadata={
        "help": "Min predicted target length when generation, excluding the source part."})
    do_sample: Optional[bool] = field(default=None, metadata={"help": "Whether to use sampling in decoding."})
    temperature: Optional[float] = field(default=0.7,
                                         metadata={"help": "The value to modulate the next token probabilities."})
    penalty_alpha: Optional[float] = field(default=0.6, metadata={
        "help": "Balance the model confidence and the degeneration penalty in contrastive search decoding"})
    top_k: Optional[int] = field(default=50, metadata={
        "help": "The number of highest probability vocabulary tokens to keep for top-k-filtering."})
    top_p: Optional[float] = field(default=0.92, metadata={
        "help": "most probable tokens with probabilities that add up to top_p or higher are kept for generation"})
    num_beams: Optional[int] = field(default=5, metadata={"help": "The beam size."})
    repetition_penalty: Optional[float] = field(default=0.6, metadata={"help": "The parameter for repetition penalty."})
    no_repeat_ngram_size: Optional[int] = field(default=3,
                                                metadata={"help": "All ngrams of that size can only occur once."})


@dataclass
class DataArguments:
    dataset_path: str = field(default=None, metadata={"help": "Path to the training dataset."})
    dataset_name: str = field(default=None, metadata={"help": "Name of the training dataset."})
    num_training_samples: int = field(default=-1, metadata={"help": "Number of training samples."})
    keep_split: int = field(default=-1, metadata={"help": "Which split is exclued in training."})
    num_val_samples: int = field(default=-1, metadata={"help": "Number of valuation samples."})
    num_test_samples: int = field(default=512, metadata={"help": "Number of test samples."})


@dataclass
class TrainingArguments(TrainingArguments):
    output_file: Optional[str] = field(default="generated_responses",
                                       metadata={"help": "The file name of generated summaries."})
    project_name: Optional[str] = field(default="FGFT", metadata={"help": "Project name for wandb logging."})