import logging
import os
from typing import List

from examples.run_classifier import NLPCC2018Processor, convert_examples_to_features, InputExample
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import numpy as np
from torch import nn
import pandas as pd

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
model_state_dict = torch.load(output_model_file, map_location='cpu')
model = BertForSequenceClassification.from_pretrained(BERT_MODEL, state_dict=model_state_dict, num_labels=len(label_list), multi_label=True)
model.to(device)


def eval_and_predict(examples: List[InputExample], multi_label, batch_size=8, eval=True):
    features = convert_examples_to_features(examples, label_list, MAX_SEQ_LENGTH, tokenizer,
                                                 multi_label=multi_label)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features],
                                 dtype=torch.float if multi_label else torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

    model.eval()
    eval_loss = 0
    nb_eval_steps, nb_eval_examples = 0, 0
    all_logits = []
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        all_logits.append(logits)
        label_ids = label_ids.to('cpu').numpy()

        eval_loss += tmp_eval_loss.mean().item()

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps

    result = {'eval_loss': eval_loss,}

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    # # Write probabilities to file
    # print("Writing probabilities to {}:".format(output_probs_file))
    all_logits = np.concatenate(all_logits, axis=0)
    if multi_label:
        all_probs = nn.Sigmoid()(torch.tensor(all_logits))
    else:
        all_probs = nn.Softmax()(torch.tensor(all_logits))
    all_probs_as_df = pd.DataFrame(data=all_probs.numpy(), columns=label_list)
    # all_probs_as_df.to_csv(output_probs_file, sep='\t')

    return all_probs_as_df


def pred_from_thresholds(prob, thresholds):
    thresholds_mask = np.tile(thresholds, reps=(prob.shape[0], 1))
    return np.greater(prob, thresholds_mask).astype(int)


# dev_prob = eval_and_predict(processor.get_dev_examples(DATA_DIR), True)
# test_prob = eval_and_predict(processor.get_test_examples(DATA_DIR), True)


demo_sentences = [
    "可是 ， 这么 惬意 的 地方 ， 出现 了 这样 大煞风景 的 一幕 ? 唉 [ 害怕 ] [ 害怕 ] [ 害怕 ]",
    "后面 才 是 亮点 … … @ BAAKOLAM   你 的 S3 借 我 耍耍 [ 泪 ]",
    "因为 梦想 ， 所以 坚持   ! [ 心 ] / / @ 羊脂球 小姐 的 爱 : .... 这 妞儿   真 美 [ 花心 ]",
]

emotion_map = {i: label for i, label in enumerate(label_list)}

demo_examples = [InputExample(i, sent) for i, sent in enumerate(demo_sentences)]
demo_prob = eval_and_predict(demo_examples, True)
demo_pred = pred_from_thresholds(demo_prob.values, [0.5]*len(label_list))
demo_pred_labels = [[emotion_map[i] for i in np.nonzero(p)[0]] for p in demo_pred]
print("Predicted emotions:")
for i, sent in enumerate(demo_sentences):
    print(f"{sent}\temotions: {demo_pred_labels[i]}")

pass
