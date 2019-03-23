import os
import gzip
import urllib.request
import numpy as np
import _pickle as pickle

filepath = 'images/mnist/mnist.pkl.gz'

def mnist_generator(data, batch_size, n_labelled, limit=None):
    if batch_size % 25 != 0:
        batch_size = 50
    images, targets = data
    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(targets)
    if limit is not None:
        print ("WARNING ONLY FIRST {} MNIST DIGITS".format(limit))
        images = images.astype('float32')[:limit]
        targets = targets.astype('int32')[:limit]
    if n_labelled is not None:
        labelled = np.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(targets)

        if n_labelled is not None:
            np.random.set_state(rng_state)
            np.random.shuffle(labelled)

        image_batches = images.reshape(-1, batch_size, 784)
        target_batches = targets.reshape(-1, batch_size)

        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)

            for i in range(len(image_batches)):
                yield (np.copy(image_batches[i]), 
		       np.copy(target_batches[i]),
		       np.copy(labelled))

        else:
            for i in range(len(image_batches)):
                yield (np.copy(image_batches[i]))
    return get_epoch


def load(batch_size, test_batch_size, n_labelled=None):
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print ("Couldn't find MNIST dataset in "+filepath+", downloading...")
        urllib.request.urlretrieve(url, filepath)

    with gzip.open(filepath, 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f, encoding='iso-8859-1')

    return (
        mnist_generator(train_data, batch_size, n_labelled), 
        mnist_generator(dev_data, test_batch_size, n_labelled), 
        mnist_generator(test_data, test_batch_size, n_labelled)
    )
