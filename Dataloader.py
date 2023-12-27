from torch.utils.data import Dataset
from transformers import BertTokenizer


class MyDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def __getitem__(self, idx):
        text = self.data[idx]  # str
        label = self.label[idx]
        inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', max_length=128, truncation=True)
        input_ids = inputs.input_ids.squeeze(0)
        token_type_ids = inputs.token_type_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        return input_ids, token_type_ids, attention_mask, label  # 包含给定文本的输入标记ID的张量,包含给定文本的标记类型ID的张量,包含给定文本的注意力掩码的张量和标签

    def __len__(self):
        return len(self.data)
