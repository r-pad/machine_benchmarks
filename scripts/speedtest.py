#!/usr/bin/env python
# Pulled from https://github.com/wkentaro/pytorch-fcn

import argparse
import time

import numpy as np
import six

import torch
import torchvision
to_tensor = torchvision.transforms.ToTensor()

def bench_load(test_dir, gpu, times, dynamic_input=False,
               image_package = 'cv2'):
    import os
    import cv2
    filename_1 = os.path.join(test_dir, 'test1.png') 
    filename_2 = os.path.join(test_dir, 'test2.png') 
    cv2.imwrite(filename_1, (255*np.random.random((224, 224, 3)).astype(np.uint8)))
    cv2.imwrite(filename_2, (255*np.random.random((224, 224, 3)).astype(np.uint8)))
    
    torch.cuda.synchronize()
    t_start = time.time()
    
    for i in six.moves.range(times):
        if dynamic_input:
            if i % 2 == 1:
                filename = filename_1
            else:
                filename = filename_2
        else:
            filename = filename_1
        
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

    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    print('Elapsed time: %.2f [s / %d loads]' % (elapsed_time, times))
    print('Hz: %.2f [hz]' % (times / elapsed_time))

    os.remove(filename_1)
    os.remove(filename_2)


def bench_eval(model_type, gpu, times, dynamic_input=False):
    with torch.no_grad():
        torch.cuda.set_device(gpu)
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

        print('==> Testing Eval {} with PyTorch'.format(model_type))
        model.eval()
        model = model.cuda()

        if dynamic_input:
            x_data = np.random.random((1, 3, 224, 224))
            x1 = torch.autograd.Variable(torch.from_numpy(x_data).float(),
                                         ).cuda()
            x_data = np.random.random((1, 3, 224, 224))
            x2 = torch.autograd.Variable(torch.from_numpy(x_data).float(),
                                         ).cuda()
        else:
            x_data = np.random.random((1, 3, 224, 224))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--test_dir', type=str, default='.')
    parser.add_argument('--image_package', type=str, default='cv2')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--times', type=int, default=1000)
    parser.add_argument('--dynamic-input', action='store_true')
    args = parser.parse_args()

    print('==> Benchmark: model=%s gpu=%d, times=%d, dynamic_input=%s' %
          (args.model, args.gpu, args.times, args.dynamic_input))
    bench_eval(args.model, args.gpu, args.times, args.dynamic_input)
    bench_load(args.test_dir, args.gpu, args.times, args.dynamic_input, image_package = args.image_package)


if __name__ == '__main__':
    main()