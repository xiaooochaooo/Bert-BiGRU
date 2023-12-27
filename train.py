import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from textProcessing import word_split
from Dataloader import MyDataSet
from model import BiGRU
from tqdm import tqdm

parameters = {
    'epochs': 20,
    'batch_size': 32,
    'train_data_path': './Data/train.tsv',
    'lr': 5e-5,
    'weight_decay': 2e-5,
}

if __name__ in '__main__':
    train_data = pd.read_csv(parameters['train_data_path'], sep='\t')
    # train_data = train_data[:100]
    labels = train_data['label'].tolist()
    data = word_split(train_data['text_a'].tolist())
    #data = train_data['text_a'].tolist()
    dataset = MyDataSet(data, labels)
    dataloader = DataLoader(dataset, batch_size=parameters['batch_size'], shuffle=True)

    device = torch.device('cuda')
    model = BiGRU().to(device)
    print(model)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=parameters['lr'], weight_decay=parameters['weight_decay'])

    cor = 0
    i = 0
    for epoch in tqdm(range(parameters['epochs'])):
        correct = 0
        total = 0
        for input_ids, token_type_ids, attention_mask, label in dataloader:
            input_ids, token_type_ids, attention_mask, label = input_ids.to(device), token_type_ids.to(
                device), attention_mask.to(device), label.to(device)
            pred = model(input_ids, token_type_ids, attention_mask)

            loss = loss_fun(pred, label)
            # print(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            _, predicted = torch.max(pred.data, 1)
            total += len(label)
            correct += (predicted == label).sum().item()
        print(correct / total)
        if correct / total > cor:
            cor = correct / total
            i=epoch
            torch.save(model.state_dict(), 'BiGRU1.pth')
    print(i)
