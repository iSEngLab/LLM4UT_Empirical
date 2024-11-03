import logging
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from tqdm import tqdm
import json

import torch
import transformers
from datasets import load_dataset

logger = logging.getLogger(__name__)


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
    torch_dtype: torch.dtype = field(default=torch.bfloat16)
    device_map: str = field(default="auto")
    pad_token: str = field(default="<pad>")
    pad_token_id: int = field(default=1)
    eos_token: str = field(default="<|endoftext|>")
    eos_token_id: int = field(default="2")


@dataclass
class GenerationArguments:
    max_new_tokens: int = field(default=512)
    reduce_token: int = field(default=10)
    max_length: int = field(default=2046)
    num_beams: int = field(default=10)
    num_return_sequences: int = field(default=1)
    device: str = field(default="cuda")
    n_gpu: int = field(default=1)
    output_dir: str = field(default="./saved_models")
    output_file: str = field(default="pred.jsonl")
    add_pad: bool = field(default=True)


def get_dataset(data_args) -> Dict:
    data_files = {"test": data_args.test_filename}
    dataset = load_dataset("json", data_dir=data_args.data_path, data_files=data_files, cache_dir=data_args.cache_dir)
    return dataset


def build_model(model_args: "ModelArguments") -> transformers.PreTrainedModel:
    return transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        torch_dtype=model_args.torch_dtype,
        device_map=model_args.device_map,
        trust_remote_code=True,
    )


def build_tokenizer(model_args: "ModelArguments",
                    generation_args: "GenerationArguments") -> transformers.PreTrainedTokenizer:
    if generation_args.add_pad:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=generation_args.max_length,
            padding_side="left",
            use_fast=True,
            trust_remote_code=True,
            # bos_token='<s>', eos_token='</s>',  
            pad_token='<pad>'
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=generation_args.max_length,
            padding_side="left",
            use_fast=True,
            trust_remote_code=True,
        )
    # logger.info(f"tokenizer eos_token {tokenizer.eos_token}, tokenizer eos_token_id: {tokenizer.eos_token_id}")
    if tokenizer.eos_token is None:
        tokenizer.eos_token = model_args.eos_token
        tokenizer.eos_token_id = model_args.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = model_args.pad_token
        tokenizer.pad_token_id = model_args.pad_token_id
    
    logger.info(f"tokenizer eos_token {tokenizer.eos_token}, tokenizer eos_token_id: {tokenizer.eos_token_id}")
    logger.info(f"tokenizer bos_token {tokenizer.bos_token}, tokenizer bos_token_id: {tokenizer.bos_token_id}")
    logger.info(f"tokenizer pad_token {tokenizer.pad_token}, tokenizer pad_token_id: {tokenizer.pad_token_id}")

    return tokenizer


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, GenerationArguments))
    model_args, data_args, generation_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )

    # Log on each process the small summary:
    logger.warning(
        f" device: {generation_args.device}"
    )
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Generation parameters {generation_args}")

    model = build_model(model_args)
    tokenizer = build_tokenizer(model_args, generation_args)
    dataset = get_dataset(data_args=data_args)
    predict_results = []
    for idx, sample in tqdm(enumerate(dataset["test"]), total=len(dataset["test"]), desc="Generating..."):
        tmp_dict = {}
        source = sample["source"]
        inputs = tokenizer(source, return_tensors='pt', truncation=True, max_length=generation_args.max_length - generation_args.reduce_token)
        inputs_len = inputs.input_ids.shape[1]
        input_ids = inputs.input_ids.to(generation_args.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=generation_args.max_new_tokens,
                # max_length=generation_args.max_length,
                num_return_sequences=generation_args.num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=generation_args.num_beams,
                early_stopping=True
            )
        outputs = outputs[:, inputs_len:]
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
