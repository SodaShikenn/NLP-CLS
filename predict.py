from config import *
from utils import *
from model import *

if __name__ == '__main__':
    id2label, _ = get_label()

    model = torch.load(MODEL_DIR + 'model_40.pth', map_location=DEVICE, weights_only=False)
    model.to(DEVICE)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    texts = [
        '小城不大，风景如画：边境小镇室韦的蝶变之路',
        '天问一号发射两周年，传回火卫一高清影像',
        '林志颖驾驶特斯拉自撞路墩起火，车头烧成废铁',
    ]

    batch_input_ids = []
    batch_mask = []
    for text in texts:
        tokened = tokenizer(text)
        input_ids = tokened['input_ids']
        mask = tokened['attention_mask']
        if len(input_ids) < TEXT_LEN:
            pad_len = (TEXT_LEN - len(input_ids))
            input_ids += [BERT_PAD_ID] * pad_len
            mask += [0] * pad_len
        batch_input_ids.append(input_ids[:TEXT_LEN])
        batch_mask.append(mask[:TEXT_LEN])

    batch_input_ids = torch.tensor(batch_input_ids).to(DEVICE)
    batch_mask = torch.tensor(batch_mask).to(DEVICE)

    with torch.no_grad():
        pred = model(batch_input_ids, batch_mask)
        pred_ = torch.argmax(pred, dim=1)

    print([id2label[l] for l in pred_])
