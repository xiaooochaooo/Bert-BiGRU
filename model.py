import torch.nn as nn
from transformers import BertModel


class BiGRU(nn.Module):
    def __init__(self):
        super(BiGRU, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')

        self.gru = nn.GRU(input_size=768, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True,dropout=0.5)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(1024, 2)

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.bert(input_ids, token_type_ids, attention_mask).last_hidden_state
        output, _ = self.gru(output)
        # print(output.shape)
        output = output[:, -1, :]
        output = self.relu(output)
        # print('最后一层', output.shape)
        output = self.linear(output)
        return output
