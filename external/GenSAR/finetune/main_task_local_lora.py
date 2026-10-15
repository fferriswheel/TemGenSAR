"""Train one task-local LoRA and evaluate adapted and frozen task paths."""

import logging
import os
from pathlib import Path

from modules.arguments import DataArguments, LLMTrainingArguments, ModelArguments
from modules.collator import TrainDataCollator
from modules.data import load_dataset
from modules.trainer import LLMTrainer
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import HfArgumentParser, set_seed


logger = logging.getLogger(__name__)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = HfArgumentParser((ModelArguments, DataArguments, LLMTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(training_args.seed)
    if not model_args.initial_model_checkpoint:
        raise ValueError("--initial_model_checkpoint is required")
    if model_args.selective_adapter_task.lower() != "rec":
        raise ValueError("This controlled entrypoint currently supports a Rec adapter")
    if data_args.val_tasks != "SeqRecWithSrcHis":
        raise ValueError("Rec adapter checkpoint selection must use Rec validation only")

    checkpoint = model_args.initial_model_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    config = AutoConfig.from_pretrained(checkpoint)
    train_dataset = load_dataset(data_args, model_args, tokenizer, mode="train")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    base_model.resize_token_embeddings(len(tokenizer))
    model = get_peft_model(
        base_model,
        LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                module.strip()
                for module in model_args.selective_lora_target_modules.split(",")
                if module.strip()
            ],
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    collator = TrainDataCollator(data_args, model_args, tokenizer)
    val_dataset = load_dataset(data_args, model_args, tokenizer, mode="valid")
    trainer = LLMTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        processing_class=tokenizer,
        data_collator=collator,
    )
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.train()
    trainer.save_state()
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    config.save_pretrained(training_args.output_dir)

    if training_args.do_predict:
        data_args.test_tasks = "SeqRecWithSrcHis"
        data_args.test_data_sample_num = "0"
        rec_test = load_dataset(data_args, model_args, tokenizer, mode="test")
        logger.info("Evaluate Rec with the task-local adapter enabled")
        trainer.evaluate(rec_test, metric_key_prefix="test_rec_adapter")

        data_args.test_tasks = "PerSrcWithRecHis"
        data_args.test_data_sample_num = "0"
        search_test = load_dataset(data_args, model_args, tokenizer, mode="test")
        logger.info("Evaluate Search with the task-local adapter disabled")
        with model.disable_adapter():
            trainer.evaluate(search_test, metric_key_prefix="test_search_frozen")


if __name__ == "__main__":
    main()
