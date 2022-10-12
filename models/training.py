import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import copy
import random
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix





def train_model(model, dataloaders, criterion, optimizer, lr_scheduler, device, args):


    best_acc = -0.1

    for epoch in range(args.num_epoches):
        print("Epoch {}/{}".format(epoch, args.num_epoches-1));
        print('+' * 80)

        train_losses = []
        train_true_labels = []
        train_pred_labels = []

        model.train()
        for x, labels in dataloaders['train']:
            # move data to GPU
            x = x.to(device, dtype=torch.float32)
            labels = labels.to(device)

            # reset optimizer.
            optimizer.zero_grad()
            logits = model(x)
            
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()


            # obtain necessary information for displaying.
            train_losses.append(loss.item())
            train_pred_labels.append(logits.detach().cpu())
            train_true_labels.append(labels.detach().cpu())
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
        logits = model(x)

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
    print("Loss: %.4f" %(np.mean(np.array(test_losses))))
    #print("F1 score: %.4f" %(f1_score(all_true_binary, all_pred_binary)))
    print("Accuracy: %.4f " %(accuracy_score(all_true_binary, all_pred_binary)))
    print(confusion_matrix(all_true_binary, all_pred_binary))