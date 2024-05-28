import torch

def get_bg_color(color):
    if color == 'white':
        return BackgroundColor.WHITE
    elif color == 'black':
        return BackgroundColor.BLACK
    elif color == 'random':
        return BackgroundColor.RANDOM
    else:
        raise ValueError("Invalid color")

class BackgroundColor:
    WHITE = torch.tensor([1.0, 1.0, 1.0])
    BLACK = torch.tensor([0.0, 0.0, 0.0])
    RANDOM = torch.rand(3)


