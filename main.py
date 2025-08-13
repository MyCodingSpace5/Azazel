from enum import Enum

import torch
import torch.nn as nn
from torchinfo import summary

class Actions(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    SPECIAL_BUTTON1 = 5
    SPECIAL_BUTTON2 = 6
    GAME_PLAYBUTTON1 = 7
    GAME_PLAYBUTTON2 = 8
    GAME_PLAYBUTTON3 = 9
    GAME_PLAYBUTTON4 = 10
    LEFT_TRIGGER = 11
    RIGHT_TRIGGER = 12
    LEFT_BUTTON = 13
    RIGHT_BUTTON = 14
def lsh_hash(storage, n_embeds: int, storage_dim):
    random_projection = torch.randn(n_embeds, storage_dim)
    projection = torch.matmul(storage, random_projection)
    return (projection > 0).int()
def fog_of_war(information_matrix, embeds):
    groups = []
    exp_power = 2 ** torch.arange(lsh_hash(information_matrix, embeds, information_matrix.dim).size(1) - 1, -1, -1)
    integer_repr = (exp_power * lsh_hash(information_matrix, embeds, information_matrix.dim)).sum(dim=1)
    remainder = torch.arange(integer_repr.size(0))
    while torch.numel(remainder) != 0:
        group_number, counts = torch.unique(remainder, return_counts=True)
        max_count_index = torch.argmax(counts)
        groups.append(group_number[max_count_index].item())
        counts_mask = remainder[max_count_index] != group_number
        remainder = remainder[counts_mask]
    return groups
class Network(nn.Module):
    def __init___(self, seq_len=None):
        super().__init__()
        self.friend = nn.Sequential(
            nn.Embedding(14, 512),
            nn.MaxPool1d(2, 2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.Linear(256 * (seq_len // 2) * 1, 128),
            nn.Linear(128, 14),
            nn.Softmax(dim=1)
        )
        self.probablity_manipulator = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * (seq_len // 2) * 1, 256)
        )
        self.enemy = nn.Sequential(
            nn.Embedding(14, 512),
            nn.MaxPool1d(2, 2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.Linear(256 * (seq_len // 2) * 1, 128),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        enemy_outputs = self.enemy(x)
        fed_outputs = fog_of_war(enemy_outputs[0], 512)
        manipulator_outputs = self.probablity_manipulator(fed_outputs)
        friend_outputs = self.friend(manipulator_outputs)
        return friend_outputs
model = Network()
def train(dataset):
    Adam = torch.optim.Adam(model.parameters(), lr=0.0000000000000000000000000000000000000000012202021, weight_decay=0.00000000000000000000000000000000000000000001, eps=0.00000000000000000000000000000000000000000000000000000000000000000000000000000191918181)
    for i, (data, targets) in enumerate(dataset):
        Adam.zero_grad()
        loss = nn.functional.kl_div(model(data), targets)
        loss.backward()
        Adam.step()
def run(x):
    return Actions(torch.argmax(model(x)))
