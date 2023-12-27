import tkinter as tk
import torch
from torch.utils.data import DataLoader
from model import BiGRU
from textProcessing import word_split
from Dataloader import MyDataSet


def on_button_click():
    s = text.get("1.0", "end-1c")  # 获取Text组件中的文本
    s = [s]
    s = word_split(s)
    dataset = MyDataSet(s, label=[0])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for input_ids, token_type_ids, attention_mask, label in dataloader:
            input_ids, token_type_ids, attention_mask, label = input_ids.to(device), token_type_ids.to(
                device), attention_mask.to(device), label.to(device)
            outputs = model(input_ids, token_type_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(probabilities, dim=1).cpu().numpy()
            if predicted[0] == 1:
                label_result.config(text='positive')
            else:
                label_result.config(text='negative')


device = torch.device('cuda')
model = BiGRU()
model.load_state_dict(torch.load('BiGRU.pth'))
model.eval()
model.to(device)
print(model)
root = tk.Tk()
root.title("情感预测")
text = tk.Text(root, width=50, height=20)
text.pack(pady=10)
button = tk.Button(root, text="预测", command=on_button_click, width=20, height=2)
button.pack()
label_result = tk.Label(root, text="", width=30, height=2, font=("Helvetica", 12))
label_result.pack(pady=10)
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_position = int((screen_width - root.winfo_reqwidth()) / 2)
y_position = int((screen_height - root.winfo_reqheight()) / 2)
root.geometry("+{}+{}".format(x_position, y_position))
root.mainloop()
