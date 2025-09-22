import functools
import inspect

from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
import torch
import random
import numpy as np

from torch import nn
from torch.nn import functional as F, MSELoss
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import Trainer, set_seed
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.trainer_pt_utils import (
    get_module_class_from_name,
)

class KDTrainer(Trainer):
    def __init__(self, teacher_model, loss_type, mean_prob=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tmp = 1
        self.teacher_model = teacher_model
        self.loss_type = loss_type

    def forward_kld(self, labels, student_logits, teacher_logits):
        mask = (labels != -100)

        model_output_log_prob = F.log_softmax(student_logits, dim=2)
        real_output_soft = F.softmax(teacher_logits / self.tmp, dim=2)

        # loss = F.kl_div(model_output_log_prob, real_output_soft, reduction="batchmean")
        kl_loss = F.kl_div(model_output_log_prob, real_output_soft, reduction="none")
        kl_loss = kl_loss.sum(-1) * mask
        kl_loss = kl_loss.sum(-1).mean()
        return kl_loss
    
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                **inputs
                # **inputs, output_hidden_states=True, output_attentions=True
            )
        teacher_logits = teacher_outputs.get("logits")
        del teacher_outputs

        # forward pass
        student_outputs = model(**inputs)
        # get attributes
        student_logits = student_outputs.get("logits")

        if not return_outputs:
            del student_outputs

        kd_loss = 0.0
        if model.kd_loss_scale > 0.0:
            if self.loss_type == "forward":
                kd_loss = self.forward_kld(inputs['labels'], student_logits, teacher_logits)

        del teacher_logits
        del student_logits

        tok_loss = model.kd_loss_scale * kd_loss
        return (tok_loss, student_outputs) if return_outputs else tok_loss
    
    