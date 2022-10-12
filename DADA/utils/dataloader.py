import random
from re import sub
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from utils.utils import normalize_data, Catergorical2OneHotCoding
from utils.augmentation import *


class MyDataset(Dataset):
    def __init__(self, filename, is_training=True, sub_policies=None, magnitudes=None, args=None):
        super(MyDataset).__init__()
        self.is_training = is_training
        self.sub_policies = sub_policies
        self.magnitudes = magnitudes
        self.args = args

        self.weights_index = None
        self.probabilities_index = None

        # first, check whether or not the file exist
        # filename_data must not be none.
        if not os.path.isfile(filename):
            print(filename + "doesn't exist!\n")
            exit(0)
        # then load the data.
        #data_dict = np.load(filename, allow_pickle=True).item();
        data = pd.read_csv(filename, sep='\t', header=None)

        # obtain the labels and convert it to one hot coding.
        #print(data.values.shape)
        self.data_y = data.values[:, 0]
        #print(self.data_y.shape)
        #print(np.min(self.data_y))
        #print(np.max(self.data_y))
        #self.data_y[self.data_y == -1] = 0
        if np.max(self.data_y) == args.num_classes:
            # change index, which starts from 1 to 0
            print("index 0") 
            self.data_y = self.data_y - 1;  # covert the label from 1:K to 0:K-1
        self.data_y = Catergorical2OneHotCoding( self.data_y.astype(np.int8))

        self.data_x = data.drop(columns=[0])
        self.data_x.columns = range(self.data_x.shape[1])
        self.data_x = self.data_x.values
        if args.data_normalization:
            std_ = self.data_x.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            self.data_x = (self.data_x - self.data_x.mean(axis=1, keepdims=True)) / std_
        self.data_x = np.expand_dims(self.data_x, axis=-1)

        #self.randaugment = RandAugment(self.args.n, self.args.m)

        #if self.is_training:
        #    mode='train'
        #else:
        #    mode = 'test'
        # load the label and convert it into one-hot coding.
        # [0, 1]-> stress; [1, 0]-> no stress
        #labels_ = np.array(np.nan_to_num(data_dict['label_'+mode]) >= 2.0) * 1
        #self.labels = np.eye(2)[labels_]
        #self.labels = np.expand_dims(labels_, axis=1).astype(np.float)
        #print(np.sum(self.labels))

    def __len__(self):
        return self.data_x.shape[0]


    def __getitem__(self, index):
        x =  self.data_x[index]
        y =  self.data_y[index]

        if self.is_training:
            magnitude = self.magnitudes.clamp(0, 1)[self.weights_index.item()]
            sub_policy = self.sub_policies[self.weights_index.item()]
            probability_index = self.probabilities_index[self.weights_index.item()]
            for i, ops_name in enumerate(sub_policy):
                if probability_index[i].item() == 1:
                    x = apply_augment(x, ops_name, magnitude[i])
        
        return x, y


def test():
    # something need to be test here.
    print("Test a function!")

if __name__ == "__main__":
    test()
    print("Everything passed")