from torch.utils import data
from config import *
import torch
from transformers import BertTokenizer
from sklearn.metrics import classification_report

from transformers import logging
logging.set_verbosity_error()

class Dataset(data.Dataset):
    def __init__(self, type='train'):
        super().__init__()
        if type == 'train':
            sample_path = TRAIN_SAMPLE_PATH
        elif type == 'dev':
            sample_path = DEV_SAMPLE_PATH
        elif type == 'test':
            sample_path = TEST_SAMPLE_PATH

        self.lines = open(sample_path).readlines() # 180,000
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        text, label = self.lines[index].split('\t')
        tokened = self.tokenizer(text)
        input_ids = tokened['input_ids']
        mask = tokened['attention_mask']
        if len(input_ids) < TEXT_LEN:
            pad_len = (TEXT_LEN - len(input_ids))
            input_ids += [BERT_PAD_ID] * pad_len
            mask += [0] * pad_len
        target = int(label)
        return torch.tensor(input_ids[:TEXT_LEN]), torch.tensor(mask[:TEXT_LEN]), torch.tensor(target)


def get_label():
    text = open(LABEL_PATH).read()
    id2label = text.split()
    return id2label, {v: k for k, v in enumerate(id2label)}


def evaluate(pred, true, target_names=None, output_dict=False):
    return classification_report(
        true,
        pred,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0,
    )


if __name__ == '__main__':
    dataset = Dataset()
    loader = data.DataLoader(dataset, batch_size=2)
    print(next(iter(loader)))