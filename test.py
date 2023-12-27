from torch.utils.data import DataLoader

from model import BiGRU
import torch
from Dataloader import MyDataSet
import pandas as pd
from textProcessing import word_split


def evaluate(model, dataloader):
    correct = 0
    total = 0
    device = torch.device('cuda')
    with torch.no_grad():
        for input_ids, token_type_ids, attention_mask, label in dataloader:
            input_ids, token_type_ids, attention_mask, label = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device), label.to(device)
            outputs = model(input_ids, token_type_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            total += len(label)
            correct += (predicted == label).sum().item()
    accuracy = correct / total
    return accuracy


if __name__ in '__main__':
    device = torch.device('cuda')
    model = BiGRU()
    model.load_state_dict(torch.load('BiGRU.pth'))
    model.eval()
    model.to(device)
    test_data = pd.read_csv('./Data/test.tsv', sep='\t')
    # test_data = test_data[:100]
    labels = test_data['label'].tolist()
    data = word_split(test_data['text_a'].tolist())

    dataset = MyDataSet(data, labels)
    test_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    test_accuracy = evaluate(model, test_loader)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
