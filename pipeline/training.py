#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from .model import BERT
from .savingandloading import *
from . import device
# from pytorchtools import EarlyStopping
from sklearn.metrics import r2_score

def modeling(model,
          optimizer,
          scheduler,
          train_loader,
          valid_loader,
          eval_every,
          file_path,
          num_epochs,
          # patience = 20,
          criterion = nn.BCELoss(),
          best_valid_loss = float("Inf")):
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    running_r2 = 0.0
    valid_running_r2 = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    train_r2_list = []
    valid_r2_list = []
    global_steps_list = []

    # early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (labels, title, content, titlecontent), _ in train_loader:
            labels = labels.type(torch.LongTensor)           
            labels = labels.to(device)
            titlecontent = titlecontent.type(torch.LongTensor)  
            titlecontent = titlecontent.to(device)
            # output = model(titlecontent, labels)
            # loss, _ = output


            optimizer.zero_grad()
            loss, output = model(titlecontent, labels)
            r2 = r2_score(labels.tolist(), torch.argmax(output, 1).tolist())
            # loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            running_r2 += r2
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    
                    # validation loop
                    for (labels, title, content, titlecontent), _ in valid_loader:
                        labels = labels.type(torch.LongTensor)
                        labels = labels.to(device)
                        titlecontent = titlecontent.type(torch.LongTensor)
                        titlecontent = titlecontent.to(device)
                        # output = model(titlecontent, labels)
                        # loss, _ = output
                        loss, output = model(titlecontent, labels)
                        r2 = r2_score(labels.tolist(), torch.argmax(output, 1).tolist())
                        # loss = criterion(output, labels)
                        
                        scheduler.step(loss)

                        valid_running_loss += loss.item()
                        valid_running_r2 += r2

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                average_train_r2 = running_r2 / eval_every
                average_valid_r2 = valid_running_r2 / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                train_r2_list.append(average_train_r2)
                valid_r2_list.append(average_valid_r2)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                running_r2 = 0.0
                valid_running_r2 = 0.0

                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}] ==> Train Loss: {:.4f}, Train R2: {:.4f}; Valid Loss: {:.4f}, Valid R2: {:.4f}; Previous LearningRate: {:.6f}'
                .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                average_train_loss, average_train_r2, average_valid_loss, average_valid_r2, scheduler.optimizer.param_groups[0]['lr']))
                # print('lr: {:.7f}'.format(scheduler.optimizer.param_groups[0]['lr']))

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'model.pt', model, best_valid_loss)
                    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')

def plot(destination_folder):
    train_loss_list, valid_loss_list, global_steps_list = load_metrics(destination_folder + '/metrics.pt')
    plt.plot(global_steps_list, train_loss_list, label='Train')
    plt.plot(global_steps_list, valid_loss_list, label='Valid')
    plt.xlabel('Global Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def training(model_path, train_iter, valid_iter, destination_folder, num_epochs, learning_rate):
    model = BERT(model_path).to(device)
#    total = sum([param.nelement() for param in model.parameters()])
#    print('total parameters: {}'.format(total))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-7)

    modeling(model, optimizer, scheduler, train_iter, valid_iter, len(train_iter) // 2, destination_folder, num_epochs)

    plot(destination_folder)

