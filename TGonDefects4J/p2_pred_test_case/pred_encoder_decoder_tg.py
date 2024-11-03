import logging
import sys
import os
import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

from tqdm import tqdm
import torch
import datasets
import transformers
import matplotlib.pyplot as plt
import numpy as np
from transformers import Trainer, Seq2SeqTrainer, EarlyStoppingCallback
from transformers.trainer import PredictionOutput
from transformers import LogitsProcessorList, MinLengthLogitsProcessor
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
        default=8,
        metadata={"help": "Number of processes to use for data preprocessing."}
    )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    torch_dtype: torch.dtype = field(default=torch.float32)
    device_map: str = field(default="auto")
    append_path: str = field(default=None)


@dataclass
class GenerationArguments:
    output_dir: str = field(default="./saved_result")
    output_file: str = field(default="pred.jsonl")
    max_new_tokens: int = field(default=512)
    max_length: int = field(default=1024)
    num_beams: int = field(default=10)
    device: str = field(default="cuda")
    n_gpu: int = field(default=1)

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        if args.get("max_new_tokens", -1) > 0:
            args.pop("max_length", None)
        else:
            args.pop("max_new_tokens", None)
        return args


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

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        torch_dtype=model_args.torch_dtype,
        # can't use "auto" in accelerate launch
        # device_map=model_args.device_map,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=generation_args.max_length,
        # padding_side="left",
        use_fast=True,
    )

    dataset = get_dataset(data_args=data_args)
    predict_results = []
    # trained_params = torch.load(model_args.append_path)
    # model.load_state_dict(trained_params)
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
                max_new_tokens=generation_args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=generation_args.num_beams,
                # early_stopping=True,
                # no_repeat_ngram_size=5,
            )
        # print(outputs)
        # outputs = outputs[:, :]
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
