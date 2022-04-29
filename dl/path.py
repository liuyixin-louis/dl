import os

def make_dir_if_not_exist(folder):
  if not os.path.exists(folder):
    os.makedirs(folder)