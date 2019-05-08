#!/usr/bin/env python
# Pulled from https://github.com/wkentaro/pytorch-fcn

import argparse
import time

import numpy as np
import six

import torch
from torch import nn, optim
import torchvision
to_tensor = torchvision.transforms.ToTensor()



def bench_load(test_dir, times, dynamic_input=False,
               image_package = 'cv2'):
    import os
    import cv2
    filename_1 = os.path.join(test_dir, 'test1.png') 
    filename_2 = os.path.join(test_dir, 'test2.png') 
    cv2.imwrite(filename_1, (255*np.random.random((224, 224, 3)).astype(np.uint8)))
    cv2.imwrite(filename_2, (255*np.random.random((224, 224, 3)).astype(np.uint8)))
    
    torch.cuda.synchronize()
    elapsed_time = 0

    for i in six.moves.range(times):
        if dynamic_input:
            if i % 2 == 1:
                filename = filename_1
            else:
                filename = filename_2
        else:
            filename = filename_1
        cv2.imwrite(filename, (255*np.random.random((224, 224, 3)).astype(np.uint8)))
        
        t_start = time.time()
        if(image_package == 'cv2'):
            img = cv2.imread(filename)
        elif(image_package == 'PIL'):
            from PIL import Image
            img = Image.open(filename)
        elif(image_package == 'skimage'):
            from skimage import io
            img = io.imread(filename)
        else:
            print('==> Invalid Image Package {}, [cv2, PIL, skimage]'.format(image_package))
        x = torch.autograd.Variable(to_tensor(img).float()).cuda()
        elapsed_time += time.time() - t_start
    print('==> Testing Load {} from {}'.format(image_package, test_dir))
    t_start = time.time()
    torch.cuda.synchronize()
    elapsed_time += time.time() - t_start

    print('Elapsed time: %.2f [s / %d loads]' % (elapsed_time, times))
    print('Hz: %.2f [hz]' % (times / elapsed_time))

    os.remove(filename_1)
    os.remove(filename_2)


def bench_eval(model_type, times, batch_size = 32, dynamic_input=False):
    with torch.no_grad():
        torch.backends.cudnn.benchmark = not dynamic_input

        if(model_type == 'resnet18'):
            model = torchvision.models.resnet18()
        elif(model_type == 'alexnet'):
            model = torchvision.models.alexnet()
        elif(model_type == 'vgg16'):
            model = torchvision.models.vgg16()
        elif(model_type == 'squeezenet'):
            model = torchvision.models.squeezenet1_0()
        elif(model_type == 'densenet'):
            model = torchvision.models.densenet161()
        elif(model_type == 'inception'):
            model = torchvision.models.inception_v3()
        elif(model_type == 'googlenet'):
            model = torchvision.models.googlenet()
        else:
            print('==> Invalid Model {}, [resnet18, alexnet, vgg16, squeezenet, densenet, inception, googlenet]'.format(model_type))
            return

        print('==> Testing Eval {}'.format(model_type))
        model.eval()
        model = model.cuda()

        if dynamic_input:

            x_data = np.random.random((batch_size, 3, 224, 224))
            x1 = torch.autograd.Variable(torch.from_numpy(x_data).float(),
                                         ).cuda()
            x_data = np.random.random((batch_size, 3, 224, 224))
            x2 = torch.autograd.Variable(torch.from_numpy(x_data).float(),
                                         ).cuda()
        else:
            x_data = np.random.random((batch_size, 3, 224, 224))
            x1 = torch.autograd.Variable(torch.from_numpy(x_data).float(),
                                         ).cuda()

        for i in six.moves.range(5):
            model(x1)
        torch.cuda.synchronize()
        t_start = time.time()
        for i in six.moves.range(times):
            if dynamic_input:
                if i % 2 == 1:
                    model(x1)
                else:
                    model(x2)
            else:
                model(x1)
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start

        print('Elapsed time: %.2f [s / %d evals]' % (elapsed_time, times))
        print('Hz: %.2f [hz]' % (times / elapsed_time))


def bench_train(model_type, times, batch_size = 32, dynamic_input=False):
    torch.backends.cudnn.benchmark = not dynamic_input

    if(model_type == 'resnet18'):
        model = torchvision.models.resnet18()
    elif(model_type == 'alexnet'):
        model = torchvision.models.alexnet()
    elif(model_type == 'vgg16'):
        model = torchvision.models.vgg16()
    elif(model_type == 'squeezenet'):
        model = torchvision.models.squeezenet1_0()
    elif(model_type == 'densenet'):
        model = torchvision.models.densenet161()
    elif(model_type == 'inception'):
        model = torchvision.models.inception_v3()
    elif(model_type == 'googlenet'):
        model = torchvision.models.googlenet()
    else:
        print('==> Invalid Model {}, [resnet18, alexnet, vgg16, squeezenet, densenet, inception, googlenet]'.format(model_type))
        return

    print('==> Testing Train {}'.format(model_type))
    model.train()
    model = model.cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    if dynamic_input:
        x_data = np.random.random((batch_size, 3, 224, 224))
        x1 = torch.autograd.Variable(torch.from_numpy(x_data).float()).cuda()
        x_data = np.random.random((batch_size, 3, 224, 224))
        x2 = torch.autograd.Variable(torch.from_numpy(x_data).float()).cuda()
        y1 = torch.autograd.Variable(torch.FloatTensor(batch_size).uniform_(0, 1000).long()).cuda()
        y2 = torch.autograd.Variable(torch.FloatTensor(batch_size).uniform_(0, 1000).long()).cuda()

    else:
        x_data = np.random.random((batch_size, 3, 224, 224))
        x1 = torch.autograd.Variable(torch.from_numpy(x_data).float()).cuda()
        y1 = torch.autograd.Variable(torch.FloatTensor(batch_size).uniform_(0, 1000).long()).cuda()


    for i in six.moves.range(5):
        model(x1)
    torch.cuda.synchronize()
    t_start = time.time()
    for i in six.moves.range(times):
        if dynamic_input:
            if i % 2 == 1:
                outputs = model(x1)
                loss = loss_function(outputs, y1)
            else:
                outputs = model(x2)
                loss = loss_function(outputs, y2)
        else:
            outputs = model(x1)
            loss = loss_function(outputs, y1)

        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    print('Elapsed time: %.2f [s / %d train]' % (elapsed_time, times))
    print('Hz: %.2f [hz]' % (times / elapsed_time))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--test-dir', type=str, default='.')
    parser.add_argument('--image-package', type=str, default='cv2')
    parser.add_argument('--times', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--dynamic-input', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()

    if(args.eval):
        bench_eval(args.model, args.times, args.batch_size, args.dynamic_input)
    if(args.train):
        bench_train(args.model, args.times, args.batch_size, args.dynamic_input)
    if(args.load):
        bench_load(args.test_dir, args.times, args.dynamic_input, image_package = args.image_package)


if __name__ == '__main__':
    main()
