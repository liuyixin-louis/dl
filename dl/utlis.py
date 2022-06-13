import torch
import random
import numpy as np
import torch
import tarfile

def get_name_from_arvix(url):
    import requests
    from bs4 import  BeautifulSoup
    # test="https://arxiv.org/abs/2202.11203"
    res = BeautifulSoup(requests.get(url).content, 'lxml').find("h1",attrs={"class":"title mathjax"})
    title = res.text[6:].replace(" ","-")
    return title


def get_timestamp():
    import datetime
    import pandas as pd
    ts = pd.to_datetime(str(datetime.datetime.now()))
    d = ts.strftime('%Y-%m-%d-%H-%M-%S')
    return d

def archive_dir(dir_name,output_filename,format="zip"):
    import shutil
    shutil.make_archive(output_filename, format, dir_name)


def untar(fname, dirs):
    """
    解压tar.gz文件
    :param fname: 压缩文件名
    :param dirs: 解压后的存放路径
    :return: bool
    """

    try:
        t = tarfile.open(fname)
        t.extractall(path = dirs)
        return True
    except Exception as e:
        print(e)
        return False


def unzip(path_to_zip_file,directory_to_extract_to):
    import zipfile
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

def intersection_of_two_tensor(t1, t2):
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    # difference = uniques[counts == 1]
    intersection = uniques[counts > 1]
    return intersection

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