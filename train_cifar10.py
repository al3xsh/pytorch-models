"""
train_cifar10.py

much of this training code is heavily influenced by the imagenet training 
code in pytorch/examples

author:     Alex Shenfield
date:       20/11/2017

"""

# imports

# import some basic os level stuff
import shutil
import time

# import pytorch stuff
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

# import my version of resnet-1001 and resnet110
from resnet.preactivation_resnet import resnet1001
from resnet.preactivation_resnet import resnet110

#
# set up some functions to train for an epoch and then run on validation data
#

# train for a single epoch
def train(epoch, train_loader, model, criterion, optimiser):
        
    # set up some measurements
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch the network to training mode
    model.train()

    # get the start time
    start = time.time()

    # get the data one batch at a time
    for batch_ix, (input, target) in enumerate(train_loader):
        
        # update the data loading time
        data_time.update(time.time() - start)
        
        # get the targets and inputs for the model (and cudarise them ...)
        input = input.cuda(async=True)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        # compute output and loss (forward pass)
        output = model(input_var)
        loss   = criterion(output, target_var)
        
        # calculate accuracy        
        precision = accuracy(output.data, target, topk=(1,5))
        
        # record accuracy and loss
        losses.update(loss.data[0],  input.size(0))
        top1.update(precision[0][0], input.size(0))
        top5.update(precision[1][0], input.size(0))
        
        # compute gradients from the backwards pass and do the optimisation 
        # step
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        # update the elapsed time
        batch_time.update(time.time() - start)
        start = time.time()
        
        # show some progress (after every batch)
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Precision@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Precision@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, batch_ix, len(train_loader), 
                      batch_time=batch_time, 
                      data_time=data_time, 
                      loss=losses, 
                      top1=top1, 
                      top5=top5))
        

# validate on the unseen validation data (I am trying to get away from the 
# train and test descriptions as testing should be completely unseen data 
# that isn't used anywhere for model selection
#
# see http://www.fast.ai/2017/11/13/validation-sets/ for a compelling argument
# as to why current practice is - often - rubbish)
def validate(val_loader, model, criterion):
        
    # set up some measurements
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch the network to training mode
    model.eval()

    # get the start time
    start = time.time()

    # get the data one batch at a time
    for batch_ix, (input, target) in enumerate(val_loader):
        
        # get the targets and inputs for the model (and cudarise them ...)
        input = input.cuda(async=True)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        # compute output and loss
        output = model(input_var)
        loss   = criterion(output, target_var)
        
        # calculate accuracy        
        precision = accuracy(output.data, target, topk=(1,5))
        
        # record accuracy and loss
        losses.update(loss.data[0],  input.size(0))
        top1.update(precision[0][0], input.size(0))
        top5.update(precision[1][0], input.size(0))
        
        # update the elapsed time
        batch_time.update(time.time() - start)
        start = time.time()
        
        # show some progress (after every batch)
        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Precision@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Precision@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      batch_ix, len(val_loader), 
                      batch_time=batch_time,
                      loss=losses, 
                      top1=top1, 
                      top5=top5))
    
    # display overall testing results
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg
    

#
# miscellaneous functions for saving data, recording data, and adjusting 
# the learning rate
#

# checkpoint the training progress and save the best model parameters found
# so far
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')    


# adjust the learning rate accoridng to the supplied schedule (i.e. use the 
# gamma variable to decay the learning rate)
def adjust_learning_rate(optimiser, epoch, schedule, gamma):
    
    # every time the current epoch is found in the schedule, decay the 
    # learning rate by gamma
    if epoch in schedule:
        for param_group in optimiser.param_groups:
            param_group['lr'] = param_group['lr'] * gamma
            
            # print an update so we can check all is well
            print('epoch = {0}, lr = {1}'.format(epoch, param_group['lr']))


# compute accuracy (or - more correctly - precision @ k)
def accuracy(output, target, topk=(1,)):
    
    # get the largest value of the topk tuple, and the size of the batch we 
    # are calculating accuracy for 
    maxk = max(topk)
    batch_size = target.size(0)

    # get the predicted class and the ground truth class
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # calculate how well we are doing with respect to every k supplied in the 
    # "topk" input 
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# define our AverageMeter class for recording training / testing data
class AverageMeter(object):
    
    # every time we create a new AverageMeter object then zero the data
    def __init__(self):
        self.reset()

    # zero the data
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    # update the data - keeping a running total and average
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


#
# main code
#
    
# this is the main part of the code where we run everything!
def main():

    # set up initial parameters
    
    # training parameters
    max_epochs = 200
    batch_size = 64
    
    best_accuracy = 0
    
    # optimisation parameters for optimiser (sgd)
    lr = 0.1
    momentum = 0.9
    weight_decay = 0.0001   # nb - this is l2 reqularisation ...
    schedule = [80, 120]
    gamma = 0.1
    
    # load the data set
    
    # print progress
    print("loading dataset - cifar10")
    
    # get the per pixel means and standard deviation for each channel
    # (precalculated - i assume on the training data ...)
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    # define the train and test transforms
    
    # for training we we flip, crop, and subtract the per pixel mean
    # (i think these are whats listed in the paper)
    train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), 
             transforms.RandomCrop(32, padding=4), 
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
    
    # for the test data we just subtract the mean
    test_transform = transforms.Compose(
            [transforms.ToTensor(), 
             transforms.Normalize(mean, std)])
    
    # get the dataset (split into test and train)
    train_data = dset.CIFAR10(root='./data', train=True, 
                              transform=train_transform, download=True)
    test_data  = dset.CIFAR10(root='./data', train=False, 
                             transform=test_transform, download=True)
    
     # define some loaders for our data
    train_loader = torch.utils.data.DataLoader(train_data, 
                                               batch_size=batch_size, 
                                               shuffle=True,
                                               num_workers=2, 
                                               pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(test_data, 
                                               batch_size=batch_size, 
                                               shuffle=False,
                                               num_workers=2, 
                                               pin_memory=True)
    
    # create model
    
    # choose which model to use!
    model = 'resnet 110'
    
    # create the appropriate model ...
    if model == 'resnet 1001':
    
        # print progress
        print("creating resnet 1001 model")
           
        # create the network and cuda it ...
        net = resnet1001()
        net.cuda()
        
        # assign it to run on all available gpus
        net = torch.nn.DataParallel(net)    
        
    elif model == 'resnet 110':
    
        # print progress
        print("creating resnet 110 model")
           
        # create the network and cuda it ...
        net = resnet110()
        net.cuda()
    
    # set up the optimisers, etc. for cifar 10
    
    # enable benchmarking in cudnn to allow autotuning of the algorithms 
    # (this can speed up the runtime - as long as the model inputs remain the
    # same size)
    torch.backends.cudnn.benchmark = True
    
    # define loss function (criterion) and optimiser
    loss_function = torch.nn.CrossEntropyLoss().cuda()
    sgd_optimiser = torch.optim.SGD(net.parameters(), lr, 
                                    momentum=momentum, 
                                    weight_decay=weight_decay, 
                                    nesterov=True)
    
    # train the network
    print("training resnet model")
    
    # train the network
    for epoch in range(max_epochs):
        
        # adjust the learning rate of the optimiser according to the 
        # learning rate schedule
        adjust_learning_rate(sgd_optimiser, epoch, schedule, gamma)
        
        # train for a single epoch
        train(epoch, train_loader, 
              model=net, criterion=loss_function, optimiser=sgd_optimiser)
        
        # evaluate on th validation set
        accuracy = validate(val_loader, model=net, criterion=loss_function)
        
        # remember the best accuracy (precision@1) and save the model 
        # checkpoint
        is_best = accuracy > best_accuracy
        best_accuracy = max(accuracy, best_accuracy)
        save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'resnet110',
                'best_accuracy': best_accuracy,
                'optimiser_state': sgd_optimiser.state_dict(),
                'model_state': net.state_dict()
                }, is_best)
        
    # and that's all she wrote :)    
    print('finished training')
    print('best result = {}'.format(best_accuracy))
    
#
# program entry point
#
if __name__ == '__main__':
    main()