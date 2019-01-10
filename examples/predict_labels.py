import logging
import os
from typing import List, Sequence, Dict

from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy

from examples.run_classifier import NLPCC2018Processor, convert_examples_to_features, InputExample
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertModel, BertConfig
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import numpy as np
from torch import nn
import pandas as pd

import allennlp.models
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor


DATA_DIR = "../data/nlpcc2018"
EVAL_BATCH_SIZE = 8
PREDICT_BATCH_SIZE = 8
OUTPUT_DIR = "../results"
MAX_SEQ_LENGTH = 128
BERT_MODEL = 'bert-base-chinese'
LOCAL_RANK = -1
NO_CUDA = True
FP16 = False

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

processor = NLPCC2018Processor()
label_list = processor.get_labels()
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)

# Setup CUDA
if LOCAL_RANK == -1 or NO_CUDA:
    device = torch.device("cuda" if torch.cuda.is_available() and not NO_CUDA else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device("cuda", LOCAL_RANK)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')

logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    device, n_gpu, bool(LOCAL_RANK != -1), FP16))


class BertForAllenNLP(BertModel, allennlp.models.Model):
    def __init__(self, config: BertConfig, num_labels: int = 2, multi_label: bool = False):
        super(BertForAllenNLP, self).__init__(config)
        self.num_labels = num_labels
        self.multi_label = multi_label
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

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
            output_dict["loss"] = loss
        else:
            return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # TODO: Implement multi-label metrics.
        #   (either ranking-based or label-based: if label-based, need to do logits -> predictions)
        return {}


# Load a trained model that you have fine-tuned
output_model_file = os.path.join(OUTPUT_DIR, "pytorch_model.bin")
model_state_dict = torch.load(output_model_file, map_location='cpu')  # Modify if running on GPU
model = BertForAllenNLP.from_pretrained(BERT_MODEL, state_dict=model_state_dict,
                                        num_labels=len(label_list), multi_label=True)
model.to(device)


def pred_from_thresholds(prob, thresholds):
    thresholds_mask = np.tile(thresholds, reps=(prob.shape[0], 1))
    return np.greater(prob, thresholds_mask).astype(int)


# demo_sentences = [
#     "可是 ， 这么 惬意 的 地方 ， 出现 了 这样 大煞风景 的 一幕 ? 唉 [ 害怕 ] [ 害怕 ] [ 害怕 ]",
#     "后面 才 是 亮点 … … @ BAAKOLAM   你 的 S3 借 我 耍耍 [ 泪 ]",
#     "因为 梦想 ， 所以 坚持   ! [ 心 ] / / @ 羊脂球 小姐 的 爱 : .... 这 妞儿   真 美 [ 花心 ]",
# ]
#
# emotion_map = {i: label for i, label in enumerate(label_list)}
#
# demo_examples = [InputExample(i, sent) for i, sent in enumerate(demo_sentences)]
# demo_prob = eval_and_predict_skorch(demo_examples, True)
# # Assuming that class imbalance was compensated for in training stage, 0.5 is an acceptable global threshold
# demo_pred = pred_from_thresholds(demo_prob, [0.5]*len(label_list))
# demo_pred_labels = [[emotion_map[i] for i in np.nonzero(p)[0]] for p in demo_pred]
# print("Predicted emotions:")
# for i, sent in enumerate(demo_sentences):
#     print(f"{sent}\temotions: {demo_pred_labels[i]}")

