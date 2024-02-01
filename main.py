# from multipleWellPlatesSmall2 import generate_fish_in_one_well
from programs.Config import Config
from programs.Aquarium import Aquarium
import shutil
import os
import os
import shutil
import multiprocessing
from multiprocessing import  Pool,cpu_count
import numpy as np
import time

homepath = Config.dataDirectory

if not os.path.exists(homepath[:-1]):
  os.makedirs(homepath[:-1])
# else:
#   # reset it
#   shutil.rmtree(homepath)
#   os.makedirs(homepath[:-1])

folders = ['images','coor_2d']
subFolders = ['train','val']
for folder in folders:
  subPath = homepath + folder
  if not os.path.exists(subPath):
      os.makedirs(subPath)

def create_and_save_data(frame_idx):
    aquarium = Aquarium(frame_idx)
    aquarium.draw()
    aquarium.saveImage()
    aquarium.saveAnnotations()

def init_pool_process():
    np.random.seed()

if __name__ == '__main__':
    # multiprocessing case
    print('Process Starting')
    startTime = time.time()
    amount = 1
    pool_obj = multiprocessing.Pool(initializer=init_pool_process)
    pool_obj.map(create_and_save_data, range(0,amount))
    pool_obj.close()
    endTime = time.time()

    print('Finish Running')
    print('Average Time: ' + str((endTime - startTime)/amount))
