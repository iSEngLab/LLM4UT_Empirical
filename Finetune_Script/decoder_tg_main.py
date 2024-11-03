import logging
import sys
import os
import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import numpy as np

import torch
import datasets
import transformers
import matplotlib.pyplot as plt
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq, EarlyStoppingCallback
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
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "Whether or not to ignore the tokens corresponding to padded labels in the loss computation."}
    )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    torch_dtype: torch.dtype = field(default=torch.float32)
    dtype: str = field(default=None)
    device_map: str = field(default="auto")
    pad_token: str = field(default="<pad>")
    pad_token_id: int = field(default=1)
    eos_token: str = field(default="<|endoftext|>")
    eos_token_id: int = field(default="2")
    use_deepspeed: bool = field(default=False)
    def __post_init__(self):
        if self.dtype == "float32":
            self.torch_dtype = torch.float32
        elif self.dtype == "float16":
            self.torch_dtype = torch.float16
        elif self.dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            pass


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    output_dir: str = field(default="./saved_models")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={"help": "The maximum input sequence length after tokenization."}
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
    add_pad: bool = field(default=False)
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
    request_num: int = field(default=1)

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


def tokenize(text, tokenizer, training_args, add_eos_token=True):
    result = tokenizer(
        text,
        truncation=True,
        max_length=training_args.model_max_length,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < training_args.model_max_length
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    if add_eos_token and len(result["input_ids"]) >= training_args.model_max_length:
        result["input_ids"][training_args.model_max_length - 1] = tokenizer.eos_token_id
        result["attention_mask"][training_args.model_max_length - 1] = 1

    result["labels"] = result["input_ids"].copy()
    return result


def get_prompt_target(sample):
    return sample['src_fm_fc_ms_ff'], sample['target']


def generate_and_tokenize_prompt(sample, tokenizer, training_args, stage: str = "train"):
    input_text, target = get_prompt_target(sample)
    full_text = input_text + target + tokenizer.eos_token
    tokenized_full_text = tokenize(full_text, tokenizer, training_args)
    tokenized_input_text = tokenize(input_text, tokenizer, training_args)
    input_len = len(tokenized_input_text["input_ids"]) - 1
    tokenized_full_text["labels"] = [IGNORE_INDEX] * input_len + tokenized_full_text["labels"][input_len:]
    return tokenized_full_text if stage != "test" else tokenized_input_text


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
            partial(generate_and_tokenize_prompt, tokenizer=tokenizer, training_args=training_args, stage="train"),
            num_proc=data_args.num_proc
        )
        result["train_dataset"] = train_dataset
    if data_args.eval_filename is not None:
        eval_dataset = dataset["eval"]
        eval_dataset = eval_dataset.map(
            partial(generate_and_tokenize_prompt, tokenizer=tokenizer, training_args=training_args, stage="eval"),
            num_proc=data_args.num_proc
        )
        result["eval_dataset"] = eval_dataset
    if data_args.test_filename is not None:
        test_dataset = dataset["test"]
        test_dataset = test_dataset.map(
            partial(generate_and_tokenize_prompt, tokenizer=tokenizer, training_args=training_args, stage="test"),
            num_proc=data_args.num_proc
        )
        result["test_dataset"] = test_dataset

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )
    result["data_collator"] = data_collator
    return result


def build_model(model_args: "ModelArguments") -> transformers.PreTrainedModel:
    if model_args.use_deepspeed is False:
        return transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            torch_dtype=model_args.torch_dtype,
            device_map=model_args.device_map,
            trust_remote_code=True
        )
    else:
        return transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            torch_dtype=model_args.torch_dtype,
            # device_map=model_args.device_map,
            trust_remote_code=True
        )


def build_tokenizer(model_args: "ModelArguments",
                    training_args: "TrainingArguments") -> transformers.PreTrainedTokenizer:
    if training_args.add_pad:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
            # bos_token='<s>', eos_token='</s>',  
            pad_token='<pad>'
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
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

    model = build_model(model_args)
    # model.parallelize()
    # model.to(training_args.device)
    # for name, param in model.named_parameters():
    #     print(name, param.shape)

    if model_args.use_deepspeed:
        import deepspeed
        model, optimizer, _, _ = deepspeed.initialize(model=model, config_params="/efs_data/sy/LLM4UT-new/LLM4UT-master/deepspeed/z3_detail.json")

    tokenizer = build_tokenizer(model_args, training_args)
    for i in model.named_parameters():
        logger.info(f"{i[0]} -> {i[1].device}")
    # for name, param in model.named_parameters():
    #     print(name, param.shape)

    data_module = get_data_module(tokenizer=tokenizer, training_args=training_args, data_args=data_args)
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stop_patience)],
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"]
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
        if training_args.do_train:
            # free memory
            del model
            del tokenizer
            # Using model after training
            model_args.model_name_or_path = training_args.output_dir
            model = build_model(model_args)
            tokenizer = build_tokenizer(model_args, training_args)

        tokenizer.padding_side = "left"  # use left-padding in generation

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # model.to(device)
        predict_results = []
        accu = []
        for idx, sample in tqdm(enumerate(data_module["test_dataset"]), total=len(data_module["test_dataset"]),
                                desc="Generating..."):
            tmp_dict = {}
            source = sample["src_fm_fc_ms_ff"]

            inputs = tokenizer(source, return_tensors='pt', truncation=True, max_length=generation_args.max_length // 4 * 3)
            inputs_len = inputs.input_ids.shape[1]
            input_ids = inputs.input_ids.to(training_args.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    # max_new_tokens=generation_args.max_new_tokens,
                    max_length=generation_args.max_length,
                    num_return_sequences=generation_args.request_num,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_beams=generation_args.num_beams
                )
            output_ids = outputs[:, inputs_len:]

            output_diff = tokenizer.batch_decode(output_ids, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=True)

            tmp_dict['predict'] = output_diff[0] if len(output_diff) == 1 else output_diff
            tmp_dict['label'] = tokenizer.decode(tokenizer.encode(sample['target']), skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=True)
            tmp_dict['xmatch'] = tmp_dict['predict'] == tmp_dict['label']
            accu.append(tmp_dict['predict'] == tmp_dict['label'])

            predict_results.append(tmp_dict)

        with open(os.path.join(training_args.output_dir, "generated_predictions.jsonl"), "w") as generated_file:
            xmatch = round(np.mean(accu) * 100, 4)
            print("\nxmatch: " + str(xmatch) + '\n')
            for predict_result in predict_results:
                generated_file.write(f"{json.dumps(predict_result)}\n")


if __name__ == "__main__":
    main()
