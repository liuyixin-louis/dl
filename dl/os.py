import os

def makedir_if_not_exist(folder):
  if not os.path.exists(folder):
    os.makedirs(folder)