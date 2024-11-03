import logging
import sys
import os
import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

import torch
import datasets
import transformers
import matplotlib.pyplot as plt
import numpy as np
from transformers import Trainer, Seq2SeqTrainer, EarlyStoppingCallback
from transformers.trainer import PredictionOutput
from datasets import load_dataset
from functools import partial

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100
# Name of the files used for checkpointing
TRAINER_STATE_NAME = "trainer_state.json"


@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={"help": "Path to the data directory"}
    )
    train_filename: str = field(
        default=None,
        metadata={"help": "Training data filename."}
    )
    eval_filename: str = field(
        default=None,
        metadata={"help": "Evaluation data filename."}
    )
    test_filename: str = field(
        default=None,
        metadata={"help": "Test data filename."}
    )
    cache_dir: str = field(
        default="./cache",
        metadata={"help": "Directory to store cached data."}
    )
    num_proc: int = field(
        default=4,
        metadata={"help": "Number of processes to use for data preprocessing."}
    )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    torch_dtype: torch.dtype = field(default=torch.float32)
    device_map: str = field(default="auto")


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    output_dir: str = field(default="./saved_models")
    optim: str = field(default="adamw_torch")
    source_max_length: int = field(
        default=512,
        metadata={"help": "The maximum input source sequence length after tokenization."}
    )
    target_max_length: int = field(
        default=256,
        metadata={"help": "The maximum input target sequence length after tokenization."}
    )
    num_train_epochs: float = field(default=75)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=5e-5)
    lr_scheduler_type: str = field(default="cosine")
    warmup_steps: int = field(default=1000)
    logging_steps: int = field(default=1)
    evaluation_strategy: str = field(default="epoch")
    eval_steps: int = field(default=1000)
    save_strategy: str = field(default="epoch")
    save_steps: int = field(default=1000)
    report_to: str = field(default="none")
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    load_best_model_at_end: bool = field(default=True)
    save_total_limit: int = field(default=10)
    seed: int = field(default=42)
    early_stop_patience: int = field(
        default=2,
        metadata={"help": "Early stopping patience for early stopping"}
    )
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves."},
    )


@dataclass
class GenerationArguments:
    max_new_tokens: int = field(default=256)
    max_length: int = field(default=1024)
    num_beams: int = field(default=10)

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        if args.get("max_new_tokens", -1) > 0:
            args.pop("max_length", None)
        else:
            args.pop("max_new_tokens", None)
        return args


def smooth(scalars: List[float]) -> List[float]:
    r"""
    EMA implementation according to TensorBoard.
    """
    last = scalars[0]
    smoothed = list()
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_loss(save_dictionary: os.PathLike, keys: List[str] = ["loss"]) -> None:
    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in keys:
        steps, metrics = [], []
        for i in range(len(data["log_history"])):
            if key in data["log_history"][i]:
                steps.append(data["log_history"][i]["step"])
                metrics.append(data["log_history"][i][key])

        if len(metrics) == 0:
            logger.warning(f"No metric {key} to plot.")
            continue

        plt.figure()
        plt.plot(steps, metrics, color="#1f77b4", alpha=0.4, label="original")
        plt.plot(steps, smooth(metrics), color="#1f77b4", label="smoothed")
        plt.title("training {} of {}".format(key, save_dictionary))
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        figure_path = os.path.join(save_dictionary, "training_{}.png".format(key.replace("/", "_")))
        plt.savefig(figure_path, format="png", dpi=100)
        print("Figure saved at:", figure_path)


def get_prompt_target(sample):
    return sample['source'], sample['target']


def generate_and_tokenize_prompt(sample, tokenizer, training_args):
    source, target = get_prompt_target(sample)
    tokenized_source = tokenizer(source, truncation=True, max_length=training_args.source_max_length,
                                 padding='max_length', return_tensors='pt')
    tokenized_target = tokenizer(target, truncation=True, max_length=training_args.target_max_length,
                                 padding='max_length', return_tensors='pt')

    return {
        "input_ids": tokenized_source["input_ids"].squeeze(0),
        "attention_mask": tokenized_source["attention_mask"].squeeze(0),
        "labels": tokenized_target["input_ids"].squeeze(0),
    }


def get_data_module(tokenizer, training_args, data_args) -> Dict:
    data_files = {}
    if data_args.train_filename is not None:
        data_files["train"] = data_args.train_filename
    if data_args.eval_filename is not None:
        data_files["eval"] = data_args.eval_filename
    if data_args.test_filename is not None:
        data_files["test"] = data_args.test_filename

    dataset = load_dataset("json", data_dir=data_args.data_path,
                           data_files=data_files, cache_dir=data_args.cache_dir)

    result = dict()
    if data_args.train_filename is not None:
        train_dataset = dataset["train"]
        train_dataset = train_dataset.map(
            partial(generate_and_tokenize_prompt, tokenizer=tokenizer, training_args=training_args),
            num_proc=data_args.num_proc
        )
        result["train_dataset"] = train_dataset
    if data_args.eval_filename is not None:
        eval_dataset = dataset["eval"]
        eval_dataset = eval_dataset.map(
            partial(generate_and_tokenize_prompt, tokenizer=tokenizer, training_args=training_args),
            num_proc=data_args.num_proc
        )
        result["eval_dataset"] = eval_dataset
    if data_args.test_filename is not None:
        test_dataset = dataset["test"]
        test_dataset = test_dataset.map(
            partial(generate_and_tokenize_prompt, tokenizer=tokenizer, training_args=training_args),
            num_proc=data_args.num_proc
        )
        result["test_dataset"] = test_dataset

    return result


def save_predictions(trainer: "Trainer", tokenizer, predict_results: "PredictionOutput") -> None:
    r"""
    Saves model predictions to `output_dir`.

    A custom behavior that not contained in Seq2SeqTrainer.
    """
    if not trainer.is_world_process_zero():
        return

    output_prediction_file = os.path.join(trainer.args.output_dir, "generated_predictions.jsonl")
    logger.info(f"Saving prediction results to {output_prediction_file}")

    labels = np.where(
        predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, tokenizer.pad_token_id
    )
    preds = np.where(
        predict_results.predictions != IGNORE_INDEX, predict_results.predictions, tokenizer.pad_token_id
    )

    for i in range(len(preds)):
        pad_len = np.nonzero(preds[i] != trainer.tokenizer.pad_token_id)[0]
        if len(pad_len):
            preds[i] = np.concatenate(
                (preds[i][pad_len[0]:], preds[i][: pad_len[0]]), axis=-1
            )  # move pad token to last

    decoded_labels = tokenizer.batch_decode(
        labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    with open(output_prediction_file, "w", encoding="utf-8") as writer:
        res: List[str] = []
        for label, pred in zip(decoded_labels, decoded_preds):
            res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
        writer.write("\n".join(res))


def main():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, GenerationArguments))
    model_args, training_args, data_args, generation_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        torch_dtype=model_args.torch_dtype,
        # can't use "auto" in accelerate launch
        # device_map=model_args.device_map,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.source_max_length,
        padding_side="right",
        use_fast=True,
    )
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    def preprocess_logits_for_metrics(logits, labels):
        """
        Original Trainer may have a memory leak.
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = torch.argmax(logits[0], dim=-1)
        return pred_ids

    data_module = get_data_module(tokenizer=tokenizer, training_args=training_args, data_args=data_args)

    def compute_metrics(eval_predictions: transformers.EvalPrediction):
        # TODO: add BLEU and CodeBLEU metrics
        pass

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stop_patience)],
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model(output_dir=training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and training_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    if training_args.do_predict:
        gen_kwargs = generation_args.to_dict()
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

        predict_results = trainer.predict(data_module["test_dataset"], metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        save_predictions(trainer, tokenizer, predict_results)


if __name__ == "__main__":
    main()
