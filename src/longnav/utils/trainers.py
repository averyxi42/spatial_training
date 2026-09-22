import math

import torch
from torch import nn
from trl import SFTTrainer
from transformers.trainer_utils import EvalPrediction
from peft import PeftType

# Assuming this helper exists in your utils, otherwise define it:
def entropy_from_logits(logits):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1)

class PrunedSFTTrainer(SFTTrainer):
    @staticmethod
    def _is_finite_metric_value(value):
        try:
            if isinstance(value, torch.Tensor):
                value = value.item()
            return math.isfinite(float(value))
        except (RuntimeError, TypeError, ValueError):
            return False

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {}
        for key, values in self._metrics[mode].items():
            finite_values = [value for value in values if self._is_finite_metric_value(value)]
            if finite_values:
                metrics[key] = sum(finite_values) / len(finite_values)

        if mode == "eval":
            metrics = {f"eval_{key}": value for key, value in metrics.items()}

        logs = {**logs, **metrics}
        super(SFTTrainer, self).log(logs, start_time)
        self._metrics[mode].clear()

    def _get_fixed_action_template_token_ids(self, device):
        cached = getattr(self, "_fixed_action_template_token_ids", None)
        if cached is not None:
            return cached.to(device)

        tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)
        token_ids = []
        for text in ("**", "<|im_end|>"):
            token_ids.extend(tokenizer.encode(text, add_special_tokens=False))

        if not token_ids:
            return None

        fixed_ids = sorted(set(token_ids))
        self._fixed_action_template_token_ids = torch.tensor(fixed_ids, dtype=torch.long)
        return self._fixed_action_template_token_ids.to(device)

    def _get_action_token_ids(self, device):
        cached = getattr(self, "_action_token_ids", None)
        if cached is not None:
            return {name: token_ids.to(device) for name, token_ids in cached.items()}

        tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)
        action_token_ids = {}
        for action in ("forward", "stop", "right", "left"):
            token_ids = tokenizer.encode(action, add_special_tokens=False)
            if token_ids:
                action_token_ids[action] = torch.tensor(sorted(set(token_ids)), dtype=torch.long)

        self._action_token_ids = action_token_ids
        return {name: token_ids.to(device) for name, token_ids in action_token_ids.items()}
    
    def _prune_tensor(self, tensor, mask, padding_value=0):
        """
        Slices a tensor (B, L) using a boolean mask (B, L) 
        and re-pads it to (B, L_new) to match the pruned logits structure.
        """
        if tensor is None:
            return None
        return tensor[:,mask]
  
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        """
        Compute training loss and additionally compute token accuracies
        Adjusted to handle Visual Pruning mismatch between Inputs and Outputs.
        """
        mode = "train" if self.model.training else "eval"

        # 1. PREPARE INPUTS (Standard TRL logic)
        labels = inputs["labels"] if "shift_labels" not in inputs else None
        inputs["use_cache"] = False
        
        if self.args.use_liger_kernel:
            inputs["return_token_accuracy"] = True
            inputs["use_token_scaling"] = self.args.loss_type == "dft"

        # 2. RUN FORWARD PASS (Super handles the model call)
        # The model internally prunes and computes the correct loss.
        # (loss, outputs) = super(SFTTrainer,self).compute_loss(
        #     model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        # )
        # for k,v in inputs.items():
        #     try:
        #         print(f"{k}:{v.shape}")
        #     except Exception as e:
        #         print(f"failed for {k}: {e}")
        outputs = model(**inputs)
        loss = outputs['loss']
        # =========================================================================
        # CRITICAL FIX: SYNC INPUTS WITH PRUNED OUTPUTS
        # =========================================================================
        # We need the pruning mask to align inputs['labels'] and inputs['attention_mask']
        # to the pruned outputs.logits shape.
        
        # Check if model returned the mask (Your model MUST return this)
        seq_keep_mask = getattr(outputs, "seq_keep_mask", None)

        if seq_keep_mask is not None:
            # 1. Update Attention Mask (used for Entropy and Token Counting)
            # We slice it so metrics don't count dropped visual tokens.
            if "attention_mask" in inputs:
                inputs["attention_mask"] = self._prune_tensor(
                    inputs["attention_mask"], seq_keep_mask, padding_value=0
                )
            
            # 2. Update Labels (used for Accuracy)
            # -100 is standard ignore index
            if "labels" in inputs:
                inputs["labels"] = self._prune_tensor(
                    inputs["labels"], seq_keep_mask, padding_value=-100
                )
                labels = inputs["labels"] # Update local reference

            if "shift_labels" in inputs:
                 inputs["shift_labels"] = self._prune_tensor(
                    inputs["shift_labels"], seq_keep_mask, padding_value=-100
                )
        else:
            print("ERROR: no sqm")
            exit()
        # =========================================================================


        # 3. METRICS LOGIC (Now safely using pruned inputs)

        # # Compute entropy
        # if not self.args.use_liger_kernel: 
        #     with torch.no_grad():
        #         per_token_entropy = entropy_from_logits(outputs.logits)
                
        #         if (
        #             self.num_virtual_tokens > 0
        #             and model.peft_config[model.active_adapter].peft_type != PeftType.PREFIX_TUNING
        #         ):
        #             per_token_entropy = per_token_entropy[:, self.num_virtual_tokens :]
                
        #         if "attention_mask" in inputs:
        #             # Uses the PRUNED attention_mask now
        #             attention_mask = inputs["attention_mask"]
        #             # Ensure shapes match (handle edge cases where padding adds 1)
        #             min_len = min(attention_mask.shape[1], per_token_entropy.shape[1])
        #             attention_mask = attention_mask[:, :min_len]
        #             per_token_entropy = per_token_entropy[:, :min_len]
                    
        #             entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
        #         elif "position_ids" in inputs:
        #             entropy = torch.mean(per_token_entropy)
        #         else:
        #             raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
                
        #         entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
        #     self._metrics[mode]["entropy"].append(entropy)

        # # Compute Token Counts (for tokens/sec)
        # if mode == "train":
        #     if "attention_mask" in inputs:
        #         # Uses PRUNED mask -> Correctly counts only processed tokens
        #         num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
        #     elif "position_ids" in inputs:
        #         # Fallback (Might be inaccurate if position_ids weren't pruned, but less likely path)
        #         local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
        #         num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
        #     else:
        #         raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
        #     self._total_train_tokens += num_tokens_in_batch
        # self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Compute Accuracy
        if self.args.use_liger_kernel:
            pass
        else:
            with torch.no_grad():
                if "shift_labels" in inputs:
                    shift_logits = outputs.logits.contiguous()
                    shift_labels = inputs["shift_labels"]
                else:
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                if (
                    self.num_virtual_tokens > 0
                    and model.peft_config[model.active_adapter].peft_type != PeftType.PREFIX_TUNING
                ):
                    shift_logits = shift_logits[:, self.num_virtual_tokens :, :]

                # Shape Safety Check before argmax
                min_len = min(shift_logits.shape[1], shift_labels.shape[1])
                shift_logits = shift_logits[:, :min_len, :]
                shift_labels = shift_labels[:, :min_len]

                predictions = shift_logits.argmax(dim=-1)
                mask = shift_labels != -100

                fixed_template_ids = self._get_fixed_action_template_token_ids(shift_labels.device)
                if fixed_template_ids is None:
                    action_mask = mask
                else:
                    fixed_template_mask = (shift_labels[..., None] == fixed_template_ids).any(dim=-1)
                    action_mask = mask & ~fixed_template_mask

                correct_action_predictions = (predictions == shift_labels) & action_mask
                total_action_tokens = action_mask.sum()
                correct_action_tokens = correct_action_predictions.sum()

                correct_action_tokens = self.accelerator.gather_for_metrics(correct_action_tokens)
                total_action_tokens = self.accelerator.gather_for_metrics(total_action_tokens)

                total_action_sum = total_action_tokens.sum()
                action_accuracy = (
                    correct_action_tokens.sum() / total_action_sum
                ).item() if total_action_sum > 0 else 0.0

                self._metrics[mode]["mean_action_token_accuracy"].append(action_accuracy)

                action_token_ids = self._get_action_token_ids(shift_labels.device)
                for action_name, action_ids in action_token_ids.items():
                    per_action_mask = action_mask & (shift_labels[..., None] == action_ids).any(dim=-1)
                    correct_per_action_tokens = ((predictions == shift_labels) & per_action_mask).sum()
                    total_per_action_tokens = per_action_mask.sum()

                    correct_per_action_tokens = self.accelerator.gather_for_metrics(correct_per_action_tokens)
                    total_per_action_tokens = self.accelerator.gather_for_metrics(total_per_action_tokens)

                    total_per_action_sum = total_per_action_tokens.sum()
                    if total_per_action_sum > 0:
                        per_action_accuracy = (
                            correct_per_action_tokens.sum() / total_per_action_sum
                        ).item()
                        self._metrics[mode][f"mean_{action_name}_action_accuracy"].append(per_action_accuracy)

        # if self.aux_loss_enabled:
        #     aux_loss = outputs.aux_loss
        #     aux_loss = self.accelerator.gather_for_metrics(aux_loss).mean().item()
        #     self._metrics[mode]["aux_loss"].append(aux_loss)

        return (loss, outputs) if return_outputs else loss
