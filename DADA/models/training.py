from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import copy
import random
from sklearn.metrics import accuracy_score, confusion_matrix

from torch.utils.tensorboard import SummaryWriter
import logging
from utils.utils import print_genotype




def print_augmentation_method(args):
    name_string=""
    if args.jitter:
        name_string += " jittering"
    elif args.scaling:
        name_string += " scaling"
    elif args.permutation:
        name_string += " permutation"
    elif args.rotation:
        name_string += " rotation"
    elif args.magwarp:
        name_string += " magwarp"
    elif args.timewarp:
        name_string += " timewarp"
    elif args.windowslice:
        name_string += " windowslice"
    elif args.windowwarp:
        name_string += " windowwarp"
    elif args.randAugment:
        name_string += " randAugment"
    else:
        pass;
    print("Backbone= "+ args.model +"; DA="+ name_string)


def train_model(model, dataloaders, criterion, optimizer, lr_scheduler, device, args):


    best_acc = -0.1

    logger = SummaryWriter()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig( 
        level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'log.txt')),
            logging.StreamHandler()
        ]
        )
    #fh = logging.FileHandler(os.path.join(args.log_dir, 'log.txt'))
    #fh.setFormatter(logging.Formatter(log_format))
    #logging.getLogger().addHandler(fh)


    #architect = Architect(model, args)

    for epoch in range(args.num_epoches):
        print("Epoch {}/{}".format(epoch, args.num_epoches-1));
        print('+' * 80)

        train_losses = []
        train_true_labels = []
        train_pred_labels = []

        logger.add_histogram("probabilities", model.probabilities, epoch)
        logger.add_histogram("magnitudes", model.magnitudes, epoch)
        logger.add_histogram("ops_weights", model.ops_weights, epoch)

        genotype = model.genotype()
        print_genotype(logging, genotype)


        model.train()

        model.sample()
        dataloaders['train'].dataset.weights_index = model.sample_ops_weights_index
        dataloaders['train'].dataset.probabilities_index = model.sample_probabilities_index

        #lr = lr_scheduler.get_lr()[0]
        for x, labels in dataloaders['train']:
            # move data to GPU
            #x = x.to(device, dtype=torch.float32)
            #labels = labels.to(device)

            x = Variable(x, requires_grad=False).to(device, dtype=torch.float32)
            labels = Variable(labels, requires_grad=False).to(device)


            #architect.step(x, labels, lr, optimizer, unrolled=args.unrolled)

            # reset optimizer.
            optimizer.zero_grad()
            logits = model(x, is_augmenting=True)
            
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()


            # obtain necessary information for displaying.
            train_losses.append(loss.item())
            train_pred_labels.append(logits.detach().cpu())
            train_true_labels.append(labels.detach().cpu())

            model.sample()
            dataloaders['train'].dataset.weights_index = model.sample_ops_weights_index
            dataloaders['train'].dataset.probabilities_index = model.sample_probabilities_index


        lr_scheduler.step()
        all_pred = np.vstack(train_pred_labels)
        all_true = np.vstack(train_true_labels)
        # convert from one-hot coding to binary label.
        all_pred_binary = np.argmax(all_pred, axis=1)
        all_true_binary = np.argmax(all_true, axis=1)
        #all_pred_binary = logits_2_binary(all_pred)
        #all_true_binary = all_true
        # output training information after each epoch.
        print("                         Training:")
        print("Loss: %.4f" %(np.mean(np.array(train_losses))))
        #F1 = f1_score(all_true_binary, all_pred_binary)
        #print("F1 score: %.4f" %(F1))
        ACC = accuracy_score(all_true_binary, all_pred_binary)
        print("Accuracy: %.4f " %(ACC))
        print(confusion_matrix(all_true_binary, all_pred_binary))
        if ACC > best_acc:
            best_acc = ACC;
            print("Save new best model")
            torch.save(model.state_dict(), os.path.join(args.model_weights_dir, 'best-model.std'))

    
    torch.save(model.state_dict(), os.path.join(args.model_weights_dir, 'model.std'))

    logger.flush()
    logger.close()

    print("#"*50)
    # Test the model
    model.load_state_dict(torch.load(os.path.join(args.model_weights_dir, 'model.std')))

    test_losses = []
    test_pred_labels = []
    test_true_labels = []
    model.eval()
    for x, labels in dataloaders['test']:
        x = x.to(device, dtype=torch.float32)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(x, is_augmenting=False)

        loss = criterion(logits,labels)

        test_losses.append(loss.item())
        test_true_labels.append(labels.detach().cpu())
        test_pred_labels.append(logits.detach().cpu())

    all_pred = np.vstack(test_pred_labels)
    all_true = np.vstack(test_true_labels)

    all_pred_binary = np.argmax(all_pred, axis=1)
    all_true_binary = np.argmax(all_true, axis=1)
    #all_pred_binary = logits_2_binary(all_pred)
    #np.save('./GSR.npy', all_pred_binary)
    #all_true_binary = all_true
    print("                         Testing:")
    print_augmentation_method(args);
    print("Loss: %.4f" %(np.mean(np.array(test_losses))))
    #print("F1 score: %.4f" %(f1_score(all_true_binary, all_pred_binary)))
    print("Accuracy: %.4f " %(accuracy_score(all_true_binary, all_pred_binary)))
    print(confusion_matrix(all_true_binary, all_pred_binary))