import json
import logging
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.trainer import Trainer

from .arguments import DataArguments, ModelArguments
from .collator import TestDataCollator
from .data import BaseDataset
from .utils import get_metrics_results, get_topk_results

logger = logging.getLogger(__name__)


class LLMTrainer(Trainer):

    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()
            self.teacher_model.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        distill_weight = inputs.pop("distill_weight", None)
        outputs = model(**inputs)
        loss = outputs.loss
        coefficient = float(getattr(self.args, "distillation_weight", 0.0))
        if self.teacher_model is not None and coefficient > 0 and distill_weight is not None:
            active = distill_weight.to(outputs.logits.device) > 0
            labels = inputs.get("labels")
            if labels is not None and bool(active.any()):
                with torch.no_grad():
                    teacher = self.teacher_model(**inputs).logits
                temperature = float(self.args.distillation_temperature)
                student_log_probs = F.log_softmax(outputs.logits / temperature, dim=-1)
                teacher_probs = F.softmax(teacher / temperature, dim=-1)
                token_kl = F.kl_div(
                    student_log_probs, teacher_probs, reduction="none"
                ).sum(dim=-1)
                valid = labels.ne(-100) & active[:, None]
                distillation = (token_kl * valid).sum() / valid.sum().clamp_min(1)
                loss = loss + coefficient * (temperature ** 2) * distillation
        return (loss, outputs) if return_outputs else loss

    def evaluate_dataset(self, model: DistributedDataParallel, task,
                         dataset: BaseDataset):
        """evaluate one dataset"""
        metrics = self.args.eval_metrics.split(',')

        local_rank = max(0, int(getattr(self.args, "local_process_index", 0)))
        world_size = max(1, int(getattr(self.args, "world_size", 1)))
        use_dist = dist.is_available() and dist.is_initialized() and world_size > 1
        if use_dist:
            sampler = DistributedSampler(dataset,
                                         shuffle=False,
                                         num_replicas=world_size,
                                         rank=local_rank,
                                         drop_last=False)
        else:
            sampler = SequentialSampler(dataset)

        data_args: DataArguments = dataset.data_args
        model_args: ModelArguments = dataset.model_args
        tokenizer = getattr(self, "processing_class", None) or getattr(
            self, "tokenizer", None)
        collator = TestDataCollator(data_args, model_args, tokenizer)
        data_loader = DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=collator,
            sampler=sampler,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True)

        if data_args.val_prompt_ids == "all":
            prompt_ids = range(len(dataset.prompts))
        else:
            prompt_ids = [int(_) for _ in data_args.val_prompt_ids.split(",")]

        all_prompt_results = []
        all_prompt_instance_results = []
        with torch.no_grad():
            for prompt_id in prompt_ids:
                if local_rank == 0:
                    logger.info("Start prompt: {}".format(prompt_id))

                data_loader.dataset.set_prompt(prompt_id)
                metrics_results = {}
                instance_results = []
                total = 0
                for step, batch in enumerate(tqdm(data_loader)):
                    inputs = batch[0].to(
                        'cuda' if torch.cuda.is_available() else 'cpu')
                    targets = batch[1]
                    bs = len(targets)

                    test_prefix_allowed_tokens = batch[3]
                    test_prefix_allowed_tokens_fn = dataset.get_prefix_allowed_tokens_fn(
                        allowed_tokens=test_prefix_allowed_tokens)

                    generation_model = model.module if hasattr(model, "module") else model
                    output = generation_model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=data_args.max_new_tokens,
                        prefix_allowed_tokens_fn=test_prefix_allowed_tokens_fn,
                        num_beams=data_args.num_beams,
                        num_return_sequences=data_args.num_beams,
                        output_scores=True,
                        return_dict_in_generate=True,
                        early_stopping=True,
                        do_sample=False)

                    output_ids = output["sequences"]
                    scores = output["sequences_scores"]

                    output = tokenizer.batch_decode(
                        output_ids, skip_special_tokens=True)

                    topk_res = get_topk_results(output,
                                                scores,
                                                targets,
                                                data_args.num_beams,
                                                all_items=None)

                    if use_dist:
                        bs_gather_list = [None for _ in range(world_size)]
                        dist.all_gather_object(obj=bs, object_list=bs_gather_list)
                        total += sum(bs_gather_list)
                        res_gather_list = [None for _ in range(world_size)]
                        dist.all_gather_object(obj=topk_res,
                                               object_list=res_gather_list)
                    else:
                        total += bs
                        res_gather_list = [topk_res]

                    if local_rank == 0:
                        all_device_topk_res = []
                        for ga_res in res_gather_list:
                            all_device_topk_res += ga_res
                        if data_args.save_instance_results:
                            instance_results.extend(all_device_topk_res)
                        batch_metrics_res = get_metrics_results(
                            all_device_topk_res, metrics)
                        for m, res in batch_metrics_res.items():
                            if m not in metrics_results:
                                metrics_results[m] = res
                            else:
                                metrics_results[m] += res

                        if (step + 1) % 50 == 0:
                            temp = {}
                            for m in metrics_results:
                                temp[m] = metrics_results[m] / total
                            print(temp)

                    if use_dist:
                        dist.barrier()

                if local_rank == 0:
                    for m in metrics_results:
                        metrics_results[m] = metrics_results[m] / total

                    all_prompt_results.append(metrics_results)
                    if data_args.save_instance_results:
                        all_prompt_instance_results.append(instance_results)
                    print("==============================================")
                    print("Prompt {} results: ".format(prompt_id),
                          metrics_results)
                    print("==============================================")
                    print("")

                if use_dist:
                    dist.barrier()

        if use_dist:
            dist.barrier()

        save_data = {}
        if local_rank == 0:
            mean_results = {}
            min_results = {}
            max_results = {}

            for m in metrics:
                all_res = [_[m] for _ in all_prompt_results]
                mean_results[m] = sum(all_res) / len(all_res)
                min_results[m] = min(all_res)
                max_results[m] = max(all_res)

            print("======================================================")
            print("Mean results: ", mean_results)
            print("Min results: ", min_results)
            print("Max results: ", max_results)
            print("======================================================")

            save_data["test_prompt_ids"] = data_args.val_prompt_ids
            save_data["mean_results"] = mean_results
            save_data["min_results"] = min_results
            save_data["max_results"] = max_results
            save_data["all_prompt_results"] = all_prompt_results
            if data_args.save_instance_results:
                save_data["all_prompt_instance_results"] = all_prompt_instance_results

            if dataset.mode == 'test':

                results_file = f'{task}_sample_neg-{data_args.num_negs}_beam-{data_args.num_beams}.json'

                results_path = os.path.join(self.args.output_dir, results_file)
                with open(results_path, "w") as f:
                    json.dump(save_data, f, indent=4)
                logger.info("Save file: {}".format(results_file))

        return save_data

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        override = eval_dataset is not None
        eval_dataset: Dict[
            str, BaseDataset] = eval_dataset if override else self.eval_dataset

        local_rank = max(0, int(getattr(self.args, "local_process_index", 0)))
        world_size = max(1, int(getattr(self.args, "world_size", 1)))
        if dist.is_available() and dist.is_initialized() and world_size > 1:
            model = DistributedDataParallel(self.model, device_ids=[local_rank])
        else:
            model = self.model
        model.eval()

        metric_for_best_model = f"{metric_key_prefix}_{self.args.metric_for_best_model}"
        task_metrics = {metric_for_best_model: []}
        for task, dataset in eval_dataset.items():
            cur_task_metrics = self.evaluate_dataset(model, task, dataset)

            if local_rank == 0:
                task_metrics[metric_for_best_model].append(
                    cur_task_metrics['max_results'][
                        self.args.metric_for_best_model])

        task_metrics = {k: np.mean(v).item() for k, v in task_metrics.items()}

        return task_metrics
