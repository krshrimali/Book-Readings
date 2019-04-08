from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
import os
import pickle
import tensorflow as tf


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num-self.last_block) * block_size)
        self.last_block = block_num


class Dataset:
    """
    Class Dataset to load data
    :param: path (str)
    :return: data loader
    """
    def __init__(self, path=None):
        """
        :param path: (str: default = None): path of the .tar.gz file)
        """
        self.path = path # .tar.gz path
        self.dataset_path = None # directory path
        self.link = None
        if self.path is None:
            if not os.path.isdir("cifar-10-batches-py"):
                self.link = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
                self.download_data(link=self.link)


    def download_data(self, link=None):
        dir_path = "cifar-10-batches-py"
        tar_gz_filename = "cifar-10-python.tar.gz"

        if not isfile(tar_gz_filename):
            if self.link is None:
                self.link = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Python Image Batches') as pbar:
                urlretrieve(
                    self.link,
                    tar_gz_filename,
                    pbar.hook
                )

        if not isdir(dir_path):
            with tarfile.open(tar_gz_filename) as tar:
                tar.extractall()
                tar.close()

    def load_data(self, path="cifar-10-batches-py", batch_num=5):
        """
        Helper function, loads data path
        :param path: (str: default = cifar-10-batches-py): path of directory containing data
        :param batch_num: (int: default = 5): batch number
        :return: input_features, target_labels
        """
        self.dataset_path = path

        with open(self.dataset_path + '/data_batch_' + str(batch_num), mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')

        input_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        target_labels = batch['labels']

        return input_features, target_labels
