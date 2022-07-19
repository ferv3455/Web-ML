#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/8/28 11:04
# @File     : train.py

"""
import argparse
import datetime
import json

import torch
import torchvision

from .models.lenet import LeNet, AdvancedLeNet
from .utils import pre_process


def get_data_loader(batch_size):
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='data/',
                                               train=True,
                                               transform=pre_process.data_augment_transform(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='data/',
                                              train=False,
                                              transform=pre_process.normal_transform())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_loader, test_loader


def evaluate(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model is: {} %'.format(100 * correct / total))
        return 100 * correct / total


def save_model(model, save_path='lenet.pth'):
    ckpt_dict = {
        'state_dict': model.state_dict()
    }
    torch.save(ckpt_dict, save_path)


def save_onnx(model, data, save_path='lenet.onnx'):
    torch.onnx.export(
        model,
        data,
        save_path,
        export_params=True,
        opset_version=10,
    )


def train(epochs, batch_size, learning_rate, num_classes, model='lenet', optimizer='adam'):

    # fetch data
    train_loader, test_loader = get_data_loader(batch_size)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Select model
    if model == 'lenet':
        model = LeNet(num_classes).to(device)
    else:
        model = AdvancedLeNet(num_classes).to(device)

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # start train
    total_step = len(train_loader)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):

            # get image and label
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, total_step, loss.item()))

        # evaluate after epoch train
        evaluate(model, test_loader, device)

    # save the trained model
    save_model(model, save_path='lenet.pth')
    # save_onnx(model, torch.randn(4, 1, 28, 28), save_path='lenet.onnx')
    return model


def train_with_log(file, epochs, batch_size, learning_rate, num_classes, model='lenet', optimizer='adam'):

    # fetch data
    train_loader, test_loader = get_data_loader(batch_size)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Select model
    if model == 'lenet':
        model = LeNet(num_classes).to(device)
    else:
        model = AdvancedLeNet(num_classes).to(device)

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Open file
    with open(file, 'r') as fp:
        json_obj = json.load(fp)

    json_obj['log'].append({
        'time': datetime.datetime.now().isoformat(sep=' '),
        'content': 'Start training.'
    })
    json_obj['log'].append({
        'time': datetime.datetime.now().isoformat(sep=' '),
        'content': 'Model: ' + str(model)
    })
    json_obj['log'].append({
        'time': datetime.datetime.now().isoformat(sep=' '),
        'content': 'Optimizer: ' + str(optimizer)
    })
    json_obj['log'].append({
        'time': datetime.datetime.now().isoformat(sep=' '),
        'content': 'Epochs:{}  Batch size:{}'.format(epochs, batch_size)
    })
    json_obj['log'].append({
        'time': datetime.datetime.now().isoformat(sep=' '),
        'content': 'Learning rate:{}  Number of classes:{}'.format(learning_rate, num_classes)
    })
    with open(file, 'w') as fp:
        json.dump(json_obj, fp)

    # start train
    losses = list()
    accuracy_hist = list()
    total_step = len(train_loader)
    for epoch in range(epochs):
        loss_in_epoch = 0
        count = 0
        for i, (images, labels) in enumerate(train_loader):

            # get image and label
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Save loss
            loss_in_epoch += loss.item() * outputs.shape[0]
            count += outputs.shape[0]

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, total_step, loss.item()))
                json_obj['log'].append({
                    'time': datetime.datetime.now().isoformat(sep=' '),
                    'content': 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                               .format(epoch + 1, epochs, i + 1, total_step, loss.item())
                })
                with open(file, 'w') as fp:
                    json.dump(json_obj, fp)

        losses.append(loss_in_epoch / count)

        # evaluate after epoch train
        accuracy = evaluate(model, test_loader, device)
        accuracy_hist.append(accuracy)

        json_obj['log'].append({
            'time': datetime.datetime.now().isoformat(sep=' '),
            'content': 'Test Accuracy of the model is: {} %'.format(accuracy)
        })
        with open(file, 'w') as fp:
            json.dump(json_obj, fp)

    json_obj['log'].append({
        'time': datetime.datetime.now().isoformat(sep=' '),
        'content': 'Training finished'
    })
    json_obj['losschart'] = losses
    json_obj['precchart'] = accuracy_hist
    with open(file, 'w') as fp:
        json.dump(json_obj, fp)

    # save the trained model
    # save_model(model, save_path='lenet.pth')
    # save_onnx(model, torch.randn(4, 1, 28, 28), save_path='lenet.onnx')

    # draw loss figure
    # matplotlib.pyplot.plot(range(1, len(losses) + 1), losses)
    # matplotlib.pyplot.show()
    # matplotlib.pyplot.savefig('./figures/{}-1.png'.format(task_id))

    # draw accuracy figure
    # matplotlib.pyplot.plot(range(1, len(accuracy_hist) + 1), accuracy_hist)
    # matplotlib.pyplot.show()
    # matplotlib.pyplot.savefig('./figures/{}-2.png'.format(task_id))

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_classes', type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # train(args.epochs, args.batch_size, args.lr, args.num_classes)
    # train_show_error(args.epochs, args.batch_size, args.lr, args.num_classes)
    train_with_log('1658042104.json', args.epochs,
                   args.batch_size, args.lr, args.num_classes)
