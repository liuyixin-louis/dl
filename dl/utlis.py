import torch
import random
import numpy as np
import torch


def evaluate_classify(model, loader, cpu, CE=torch.nn.CrossEntropyLoss()):
    acc = AverageMeter()
    loss = AverageMeter()

    model.eval()
    if not cpu: CE = CE.cuda()
    for x, y in loader:
        if not cpu: x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            _y = model(x)
            ac = (_y.argmax(dim=1) == y).sum().item() / len(x)
            lo = CE(_y,y).item()
        acc.update(ac, len(x))
        loss.update(lo, len(x))

    return acc.average(), loss.average()


class AverageMeter():
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        self.mean = 0

    def update(self, val, cnt):
        self.cnt += cnt
        self.sum += val * cnt
        self.mean = self.sum / self.cnt

    def average(self):
        return self.mean

    def total(self):
        return self.sum


def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators.
    :param seed:
    :param using_cuda:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_random_seed(12)
    print(np.random.rand(2))
    # print(np.random.rand(2))