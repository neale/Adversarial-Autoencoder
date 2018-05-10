import os
import gzip
import urllib
import numpy as np
import _pickle as pickle
import glob
from scipy.misc import imread

def celeba_generator(batch_size, data_dir):
    all_data = []

    paths = glob.glob(data_dir+'*.jpg')
    for fn in paths:
        all_data.append(imread(fn))
    images = np.concatenate(all_data, axis=0)

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)

        for i in range(int(len(images) / batch_size)):
            yield np.copy(images[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load(batch_size, data_dir):
    return celeba_generator(batch_size, data_dir)
