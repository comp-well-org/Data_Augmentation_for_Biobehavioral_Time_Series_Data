from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import os
import sys
import argparse
from sklearn.utils import shuffle
import torch
from models.training import train_model
from models.model import *
from utils.dataloader import MyDataset
from utils.primitives import sub_policies
from models.model_search import Network

from torch import optim
from torch.utils.data import  Dataset, DataLoader, TensorDataset
import random


parser = argparse.ArgumentParser()
    
# Augmentation   methods
#parser.add_argument('--augmentation_ratio', type=int, default=1, help="How many times to augment")
parser.add_argument('--seed',type=int,     default=10000, help="Randomization seed")
parser.add_argument('--jitter',            default=False, action="store_true", help="Jitter preset augmentation")
parser.add_argument('--scaling',           default=False, action="store_true", help="Scaling preset augmentation")
parser.add_argument('--permutation',       default=False, action="store_true", help="Equal Length Permutation preset augmentation")
parser.add_argument('--rotation',          default=False, action="store_true", help="Rotation preset augmentation")
parser.add_argument('--magwarp',           default=False, action="store_true", help="Magnitude warp preset augmentation")
parser.add_argument('--timewarp',          default=False, action="store_true", help="Time warp preset augmentation")
parser.add_argument('--windowslice',       default=False, action="store_true", help="Window slice preset augmentation")
parser.add_argument('--windowwarp',        default=False, action="store_true", help="Window warp preset augmentation")
#parser.add_argument('--spawner',           default=False, action="store_true", help="SPAWNER preset augmentation")
#parser.add_argument('--dtwwarp',           default=False, action="store_true", help="DTW warp preset augmentation")
#parser.add_argument('--shapedtwwarp',      default=False, action="store_true", help="Shape DTW warp preset augmentation")
#parser.add_argument('--wdba',              default=False, action="store_true", help="Weighted DBA preset augmentation")
#parser.add_argument('--discdtw',           default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
#parser.add_argument('--discsdtw',          default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
parser.add_argument('--randAugment',        default=False, action="store_true", help="Rand Augmentation.")

# File settings
parser.add_argument('--data_dir', type=str, default="./data", help="Data dir")
parser.add_argument('--train_data_file', type=str, default="", help="Train data file")
parser.add_argument('--test_data_file', type=str, default="", help="Test data file")
parser.add_argument('--model_weights_dir', type=str, default="./experiments", help="Model weight path")
parser.add_argument('--log_dir', type=str, default="./logs", help="Log path")
parser.add_argument('--data_normalization', default=True, action="store_true", help="Normalize data to the range [-1,1]")
parser.add_argument('--delimiter', type=str, default=" ", help="Delimiter")

# training hyper-parameters
parser.add_argument('--num_epoches',type=int,     default=25, help="training epoches")
parser.add_argument('--lr',type=float,              default=0.001, help="initial learning rate")
parser.add_argument('--decay_rate',type=float,      default=0.1, help="decay rate for learning rate")
parser.add_argument('--step_size',type=int,       default=10, help="decrease learning rate by decay_rate for every step_size")
parser.add_argument('--batch_size',type=int,      default=100, help="batch size")
parser.add_argument('--cuda_device', type=str,    default="cuda:0", help="cuda device")


# parameters for Autoaugment
parser.add_argument('--n',type=int,      default=3, help=" Number of augmentation transformations to apply sequentially.")
parser.add_argument('--m',type=int,      default=15, help=" Magnitude for all the transformations.")

# parameters for DADA.
parser.add_argument('--num_policies',type=int,      default=30, help="Number of policies to be used for DADA.")

# model settings.
# [B, C, Len]
parser.add_argument('--model', type=str, default="mlp", help="backbone model choosed in the experiments")
parser.add_argument('--input_length',type=int,    default=100, help="length of input")
parser.add_argument('--input_channels',type=int,  default=1, help="channels of input")
parser.add_argument('--num_classes',type=int,     default=2, help="number of classes")


args = parser.parse_args()

sub_policies = random.sample(sub_policies, args.num_policies) # i.e., 30 policies, each contains two operations.
print(sub_policies)


def main():

    # manully set the seed, so we can alleviate the randomness in model's performance evaluation.
    torch.manual_seed(args.seed)


    ##########################################
    ## to test the input data pipeline.
    #x, y = next(iter(dataloaders["train"]))
    #print(x.shape)
    #print(y.shape)
    #exit()
    ###########################################

    ## define model and device, and move the model to the pre-defined device.
    #if args.model == "mlp":
    #    model = model_mlp(args.input_length, args.num_classes)
    #elif args.model == "conv-1d":
    #    model = model_conv1d(args.input_length, args.input_channels, args.num_classes)
    #elif args.model == "resnet-1d":
    #    model = model_ResNet(BasicBlock, [2, 2, 2, 2], args.num_classes)
    #else:
    #    print("Undefined model type!")

    model = Network(args.model, args.input_length, args.num_classes, sub_policies)
    device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    # prepare training data and testing data.
    dataloaders={
        'train':DataLoader(MyDataset(os.path.join(args.data_dir, args.train_data_file), is_training=True, sub_policies=sub_policies,magnitudes=model.magnitudes, args=args), batch_size=args.batch_size, shuffle=True ),
        'test': DataLoader(MyDataset(os.path.join(args.data_dir, args.test_data_file), is_training=False, args=args), batch_size=args.batch_size, shuffle=False)
    }

    # define  an optimizer
    params_to_update = model.parameters()
    print("Params to learn: ")
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t", name)
    optimizer = optim.Adam(params_to_update, lr=args.lr)
    # adjust learning rate by gamma every step_size epochs.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.decay_rate)
    lr_scheduler.step();   

    # define learing rate scheduler. 


    # define loss function.
    criterion = nn.BCEWithLogitsLoss()



    train_model(model, dataloaders, criterion, optimizer, lr_scheduler, device, args);



if __name__ == "__main__":
    main()