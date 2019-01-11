from typing import Dict, Optional

import allennlp.models
import torch
from torch import nn
from allennlp.training.metrics import Metric
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy


from pytorch_pretrained_bert import BertModel, BertConfig


class BertForAllenNLP(BertModel, allennlp.models.Model):
    def __init__(self, config: BertConfig, num_labels: int = 2, multi_label: bool = False):
        super(BertForAllenNLP, self).__init__(config)
        self.num_labels = num_labels
        self.multi_label = multi_label
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

        self.hamming_loss = HammingLoss()

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None, labels: torch.Tensor = None,
                class_weight: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        output_dict = {"logits": logits}

        if labels is not None:
            if self.multi_label:
                loss_fct = binary_cross_entropy_with_logits(pos_weight=class_weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            else:
                loss_fct = cross_entropy(weight=class_weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            self.hamming_loss(torch.sigmoid(logits) > 0.5, labels)
            output_dict["loss"] = loss
        else:
            return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # TODO: Implement multi-label metrics.
        #   (either ranking-based or label-based: if label-based, need to do logits -> predictions)
        return {"hamming_loss": self.hamming_loss.get_metric(reset)}


class HammingLoss(Metric):
    def __init__(self):
        self._correct_pred = 0
        self._incorrect_pred = 0

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.Tensor] = None):
        total_pred = predictions.numpy().size
        self._correct_pred += torch.sum(predictions == gold_labels).item()
        self._incorrect_pred += total_pred - self._correct_pred

    def get_metric(self, reset: bool = False):
        hamming_loss = self._correct_pred / (self._correct_pred + self._incorrect_pred)
        if reset:
            self.reset()

    def reset(self):
        self._correct_pred = 0
        self._incorrect_pred = 0
