import os
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from optimizer import RobustOptimizer

def test(model, device, test_loader, criterion, row_all):

    model.eval()
    epoch_val_accuracy = 0
    epoch_val_loss = 0

    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().sum()
            epoch_val_accuracy += acc
            epoch_val_loss += val_loss
    epoch_val_accuracy /= row_all
    epoch_val_loss /= row_all
    return epoch_val_accuracy, epoch_val_loss

def train(model, device, train_loader, test_loader, criterion, train_row_all, test_row_all, n_epochs, learningrate, robust=False):

    # optimizer
    if robust:
        optimizer = RobustOptimizer(filter(lambda p: p.requires_grad, model.parameters()), optim.Adam, lr=learningrate)
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learningrate)

    # scheduler
    # scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    best_acc = 0
    best_model = None # save best model

    for epoch in range(n_epochs):

        epoch_loss = 0
        epoch_accuracy = 0

        model.train()

        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)

            if robust:
                # optimizer.zero_grad()
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # second forward-backward pass
                output = model(data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc = (output.argmax(dim=1) == label).float().sum()
            epoch_accuracy += acc
            epoch_loss += loss
        epoch_accuracy /= train_row_all
        epoch_loss /= train_row_all

        epoch_val_accuracy, epoch_val_loss = test(model, device, test_loader, criterion, test_row_all)
        print(
            f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
        if best_acc < epoch_val_accuracy:
            best_acc = epoch_val_accuracy
            best_model = copy.deepcopy(model.state_dict())
        # scheduler.step()

    if best_model is not None:
        model.load_state_dict(best_model)
        print(f"Best acc:{best_acc}")
        epoch_val_accuracy, epoch_val_loss = test(model)
        print(
            f"val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
    else:
        print(f"No best model Best acc:{best_acc}")
