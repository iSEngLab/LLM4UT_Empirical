import logging
import sys
import os
import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import re

import torch
import torch.nn as nn
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
    model_type: str = field(default=None)


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
    label_names: Optional[List[str]] = field(
        default_factory=lambda: ["target_ids", "target_attention_mask", "is_pred"],
        metadata={"help": "label names in Model forward"}
    )
    save_safetensors: bool = field(
        default=False
    )


@dataclass
class GenerationArguments:
    max_new_tokens: int = field(default=256)
    max_length: int = field(default=1024)
    num_beams: int = field(default=50)

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


def filter_code(codes):
    codes = codes.replace('\r',' ').replace('\n',' ').replace('\t',' ')
    codes = re.sub(' +', ' ', codes)
    return codes

def get_prompt_target(sample):
    return sample['focal_src'], sample['focal_tgt'], sample['test_src'], sample['test_tgt']


def generate_and_tokenize_prompt(sample, tokenizer, training_args, dataset_type: str = "train"):
    focal_src, focal_tgt, test_src, test_tgt = get_prompt_target(sample)
    source = [repr(focal_src)[1:-1], repr(focal_tgt)[1:-1], repr(test_src)[1:-1]]
    source_ids_list = []
    source_masks_list = []
    each_length = training_args.source_max_length // 3
    for each in source:
        ret = tokenizer(each, truncation=True, max_length=each_length,
                                padding='max_length', return_tensors='pt')
        source_ids_list.append(ret["input_ids"].squeeze(0))
        source_masks_list.append(ret["attention_mask"].squeeze(0))

    final_source_ids = torch.cat(source_ids_list)
    final_source_masks = torch.cat(source_masks_list)
    if final_source_ids.size(0) < training_args.source_max_length:
        padding_length = training_args.source_max_length - final_source_ids.size(0)
        final_source_ids = torch.cat([final_source_ids, torch.full((padding_length,),tokenizer.pad_token_id)])
        final_source_masks = torch.cat([final_source_masks, torch.zeros(padding_length)])

    # source = repr(focal_src)[1:-1] + repr(focal_tgt)[1:-1] + repr(test_src)[1:-1]
    # source = repr(test_src)[1:-1]
    target = filter_code(repr(test_tgt)[1:-1])
    # source, target = get_prompt_target(sample)
    # tokenized_source = tokenizer(source, truncation=True, max_length=training_args.source_max_length,
    #                              padding='max_length', return_tensors='pt')
    tokenized_target = tokenizer(target, truncation=True, max_length=training_args.target_max_length,
                                 padding='max_length', return_tensors='pt')
    

    # print("shape: {}".format(final_source_ids.shape))
    # print("input_ids: {}".format(final_source_ids))
    # print("attention_mask: {}".format(final_source_masks))
    # print("label: {}".format(tokenized_target["input_ids"].squeeze(0)))

    return {
        "input_ids": final_source_ids,
        "attention_mask": final_source_masks,
        "labels": tokenized_target["input_ids"].squeeze(0),
        # Custom Label Value
        # If in prediction stage, do not have label value
        "target_ids": tokenized_target["input_ids"].squeeze(0),
        "target_attention_mask": tokenized_target["attention_mask"].squeeze(0),
        "is_pred": True if dataset_type == "test" else False
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
            partial(generate_and_tokenize_prompt, tokenizer=tokenizer, training_args=training_args,
                    dataset_type="train"),
            num_proc=data_args.num_proc
        )
        result["train_dataset"] = train_dataset
    if data_args.eval_filename is not None:
        eval_dataset = dataset["eval"]
        eval_dataset = eval_dataset.map(
            partial(generate_and_tokenize_prompt, tokenizer=tokenizer, training_args=training_args,
                    dataset_type="eval"),
            num_proc=data_args.num_proc
        )
        result["eval_dataset"] = eval_dataset
    if data_args.test_filename is not None:
        test_dataset = dataset["test"]
        test_dataset = test_dataset.map(
            partial(generate_and_tokenize_prompt, tokenizer=tokenizer, training_args=training_args,
                    dataset_type="test"),
            num_proc=data_args.num_proc
        )
        result["test_dataset"] = test_dataset

    return result


def save_predictions(trainer: "Trainer", tokenizer, predict_results: "PredictionOutput", args) -> None:
    """
    Saves model predictions to `output_dir`.

    A custom behavior that not contained in Seq2SeqTrainer.
    """
    if not trainer.is_world_process_zero():
        return

    output_prediction_file = os.path.join(trainer.args.output_dir, "generated_predictions.jsonl")
    logger.info(f"Saving prediction results to {output_prediction_file}")

    # Handling multiple beams
    num_samples = predict_results.predictions.shape[0]
    num_beams = args.num_beams

    # Assuming that labels are the same for all beams
    labels = np.where(
        predict_results.label_ids[0] != IGNORE_INDEX, predict_results.label_ids[0], tokenizer.pad_token_id
    )
    decoded_labels = tokenizer.batch_decode(
        labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    with open(output_prediction_file, "w", encoding="utf-8") as writer:
        res = []
        accs = []

        # Process each sample's predictions
        for i in range(num_samples):
            best_match = ""
            best_pred = None

            # Check each beam for the best prediction
            for beam_index in range(num_beams):
                predictions = predict_results.predictions[i, beam_index, :]
                preds = np.where(predictions != IGNORE_INDEX, predictions, tokenizer.pad_token_id)

                pad_idx = np.nonzero(preds != trainer.tokenizer.pad_token_id)[0]
                if len(pad_idx):
                    preds = np.concatenate((preds[pad_idx[0]:], preds[:pad_idx[0]]), axis=-1)  # move pad token to last

                decoded_preds = tokenizer.decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                if decoded_labels[i] == decoded_preds:
                    best_match = decoded_preds
                    break
                elif best_pred is None:
                    best_pred = decoded_preds  # Keep the first prediction as the best if none match

            # Finalize the best prediction
            if best_match:
                accs.append(1)
                res.append(json.dumps({"xmatch": True, "label": decoded_labels[i], "predict": best_match}, ensure_ascii=False))
            else:
                accs.append(0)
                res.append(json.dumps({"xmatch": False, "label": decoded_labels[i], "predict": best_pred}, ensure_ascii=False))

        # Calculate and write overall match accuracy
        xmatch = round(np.mean(accs) * 100, 4)
        logger.info(f"\nXmatch : {xmatch}")
        # writer.write(f"xmatch: {xmatch}\n")
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

    # build model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.source_max_length,
        padding_side="right", # codegeex-6b is left
        use_fast=True,
        trust_remote_code=True
    )
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path,trust_remote_code=True)
    encoder = transformers.AutoModel.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        torch_dtype=model_args.torch_dtype,
        device_map=model_args.device_map,
        trust_remote_code=True
    )
    if model_args.model_type == "codebert":
        from codebert import model
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model = model.Seq2SeqModel(encoder=encoder,
                                    decoder=decoder,
                                    tokenizer=tokenizer,
                                    config=config,
                                    beam_size=generation_args.num_beams,
                                    max_length=generation_args.max_length,
                                    sos_id=tokenizer.cls_token_id,
                                    eos_id=tokenizer.sep_token_id)
    elif model_args.model_type == "graphcodebert":
        from graphcodebert import model
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model = model.Seq2SeqModel(encoder=encoder,
                                    decoder=decoder,
                                    tokenizer=tokenizer,
                                    config=config,
                                    beam_size=generation_args.num_beams,
                                    max_length=generation_args.max_length,
                                    sos_id=tokenizer.cls_token_id,
                                    eos_id=tokenizer.sep_token_id)
    elif model_args.model_type == "unixcoder":
        from unixcoder import model
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model = model.Seq2SeqModel(encoder=encoder,
                                    decoder=decoder,
                                    tokenizer=tokenizer,
                                    config=config,
                                    beam_size=generation_args.num_beams,
                                    max_length=generation_args.max_length,
                                    sos_id=tokenizer.cls_token_id,
                                    eos_id=tokenizer.sep_token_id)
    elif model_args.model_type == "codegeex2":
        model = encoder
    else:
        raise ValueError(f"do not support model type {model_args.model_type}")

    data_module = get_data_module(tokenizer=tokenizer, training_args=training_args, data_args=data_args)
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stop_patience)],
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
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
        model.load_state_dict(torch.load(training_args.output_dir + "/pytorch_model.bin"))
        gen_kwargs = generation_args.to_dict()
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

        predict_results = trainer.predict(data_module["test_dataset"], metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        save_predictions(trainer, tokenizer, predict_results, generation_args)


if __name__ == "__main__":
    main()
