# model.py
import torch
import torch.nn as nn

class CarPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),     # уменьшили с 16 до 12. 205 строк vs 31 признак. высокий риск переобучения. 
            nn.ReLU(),
            nn.Dropout(dropout_rate),             # случайно отключает 10% нейронов. Стд. защита от запоминания тренировочных данных 
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.net(x)