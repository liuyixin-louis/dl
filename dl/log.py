import time


def time_stamp():
    return time.strftime("%Y%m%d%H%M%S", time.localtime())


if __name__ == '__main__':
    print(time_stamp())
