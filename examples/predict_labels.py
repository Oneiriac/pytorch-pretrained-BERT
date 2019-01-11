import logging
import os
from typing import List, Sequence, Dict

from allennlp.predictors import SentenceTaggerPredictor
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy

from examples.run_classifier import NLPCC2018Processor, convert_examples_to_features, InputExample
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertModel, BertConfig
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import numpy as np
from torch import nn
import pandas as pd

from pytorch_pretrained_bert.allen_nlp import BertForAllenNLP

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



# Load a trained model that you have fine-tuned
output_model_file = os.path.join(OUTPUT_DIR, "pytorch_model.bin")
model_state_dict = torch.load(output_model_file, map_location='cpu')  # Modify if running on GPU
model = BertForAllenNLP.from_pretrained(BERT_MODEL, state_dict=model_state_dict,
                                        num_labels=len(label_list), multi_label=True)


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

