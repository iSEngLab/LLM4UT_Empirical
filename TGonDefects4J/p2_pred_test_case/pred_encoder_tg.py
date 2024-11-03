import logging
import sys
import os
import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

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
from tqdm import tqdm

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
    append_path: str = field(default=None)
    torch_dtype: torch.dtype = field(default=torch.float32)
    device_map: str = field(default="auto")
    model_type: str = field(default=None)

@dataclass
class GenerationArguments:
    output_dir: str = field(default="./saved_result")
    output_file: str = field(default="pred.jsonl")
    max_new_tokens: int = field(default=256)
    max_length: int = field(default=512)
    num_beams: int = field(default=1)
    device: str = field(default="cuda")
    n_gpu: int = field(default=1)

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


def get_dataset(data_args) -> Dict:
    data_files = {"test": data_args.test_filename}
    dataset = load_dataset("json", data_dir=data_args.data_path, data_files=data_files, cache_dir=data_args.cache_dir)
    return dataset


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, GenerationArguments))
    model_args, data_args, generation_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # build model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=generation_args.max_length,
        use_fast=True
    )
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    encoder = transformers.AutoModel.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        torch_dtype=model_args.torch_dtype,
        trust_remote_code=True
    )

    if model_args.model_type == "codebert":
        from codebert import model
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        decoder = decoder.to(generation_args.device)
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

    dataset = get_dataset(data_args=data_args)
    predict_results = []
    trained_params = torch.load(model_args.append_path)
    model.load_state_dict(trained_params)
    model = model.to(generation_args.device)

    for idx, sample in tqdm(enumerate(dataset["test"]), total=len(dataset["test"]), desc="Generating..."):
        tmp_dict = {}
        source = sample["source"]
        inputs = tokenizer(source, return_tensors='pt', truncation=True, max_length=generation_args.max_length)
        inputs = inputs.to(generation_args.device)
        input_ids = inputs.input_ids.to(generation_args.device)
        attention_mask = inputs.attention_mask.to(generation_args.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # max_new_tokens=generation_args.max_new_tokens,
                # pad_token_id=tokenizer.pad_token_id,
                # eos_token_id=tokenizer.eos_token_id,
                # num_beams=generation_args.num_beams,
                # logits_processor=logits_processor,
                # early_stopping=True,
                # no_repeat_ngram_size=5,
            )
        # print(outputs)
        outputs = outputs[0]
        # print(outputs)
        # output_ids = outputs[:, inputs_len:].detach().cpu().tolist()
        output_diff = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        tmp_dict['id'] = sample["id"]
        tmp_dict['revision'] = sample["revision"]
        tmp_dict['predict'] = output_diff[0] if len(output_diff) == 1 else output_diff
        predict_results.append(tmp_dict)

    if not os.path.exists(generation_args.output_dir):
        os.makedirs(generation_args.output_dir)
    output_path = os.path.join(generation_args.output_dir, generation_args.output_file)
    with open(output_path, "w", encoding="utf-8") as output_file:
        for result in predict_results:
            json.dump(result, output_file)
            output_file.write('\n')

if __name__ == "__main__":
    main()
