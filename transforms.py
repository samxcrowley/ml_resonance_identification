import torch
import torch.nn.functional as F
import torchvision.transforms
import data

def _resnet34_transform(sobel=False):

    ls = []

    ls.append(_lambda(lambda x: _normalise(x)))
    if sobel:
        ls.append(_lambda(lambda x: _sobel(x)))
    ls.append(_lambda(lambda x: x.repeat(3, 1, 1)))
    ls.append(torchvision.models.ResNet34_Weights.DEFAULT.transforms())

    transform = torchvision.transforms.Compose(ls)

    return transform

def _encoder_transform(sobel=False):

    ls = []

    ls.append(_lambda(lambda x: _normalise(x)))
    if sobel:
        ls.append(_lambda(lambda x: _sobel(x)))

    transform = torchvision.transforms.Compose(ls)

    return transform

def _lambda(foo, flag=True):
    return torchvision.transforms.Lambda(foo if flag else (lambda x: x))

def _normalise(x, min=None, max=None):

    if min == None:
        min = x.min()
    if max == None:
        max = x.max()

    return (x - min) / (max - min)

def _print_shape(x):
    print(x.shape)
    return x

def _sobel(x):

    Kx = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=torch.float32)
    Ky = torch.tensor([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=torch.float32)

    # [1, 1, W, H]
    x = x.float().unsqueeze(0).unsqueeze(0)

    # [1, 1, 3, 3]
    Kx = Kx.unsqueeze(0).unsqueeze(0)
    Ky = Ky.unsqueeze(0).unsqueeze(0)

    Gx = F.conv2d(x, Kx, padding=1)
    Gy = F.conv2d(x, Ky, padding=1)

    magnitude = torch.sqrt(Gx**2 + Gy**2)

    return magnitude.squeeze()