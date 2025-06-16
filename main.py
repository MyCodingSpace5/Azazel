import torch
import torch.nn as nn

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
class Network(nn.Module):
    def __init___(self):
        super().__init__()
        
