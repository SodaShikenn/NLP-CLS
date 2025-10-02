TRAIN_SAMPLE_PATH = './code/input/train.txt' 
DEV_SAMPLE_PATH = './code/input/dev.txt' # eval.test 
TEST_SAMPLE_PATH = './code/input/test.txt' 

LABEL_PATH = './code/input/class.txt' 

BERT_PAD_ID = 0
TEXT_LEN = 30

BERT_MODEL = '../hf_demo/bert-base-chinese'
MODEL_DIR = './code/output/models/'

EMBEDDING_DIM = 768
NUM_FILTERS = 256
NUM_CLASSES = 10
FILTER_SIZES = [2, 3, 4]

EPOCH = 200
LR = 1e-3

import torch
DEVICE = torch.device('cuda')


if __name__ == '__main__':
    print(torch.tensor([1,2,3]).to(DEVICE))