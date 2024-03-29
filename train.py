import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from torchvision import datasets, transforms, utils
from models import LabelNet, PixelCNN


n_classes = 10 # number of classes
n_epochs = 25 # number of epochs to train
n_layers = 7 # number of convolutional layers
n_channels = 16 # number of channels
device = 'cuda:0'

def to_one_hot(y, k=10):
    y = y.view(-1, 1)
    y_one_hot = torch.zeros(y.numel(), k)
    y_one_hot.scatter_(1, y, 1)
    return y_one_hot.float()

pixel_cnn = PixelCNN(n_channels, n_layers).to(device)
label_net = LabelNet().to(device)

trainloader = data.DataLoader(datasets.MNIST('data', train=True,
                                             download=True,
                                             transform=transforms.ToTensor()),
                              batch_size=128, shuffle=True,
                              num_workers=1, pin_memory=True)

testloader = data.DataLoader(datasets.MNIST('data', train=False,
                                            download=True,
                                            transform=transforms.ToTensor()),
                             batch_size=128, shuffle=False,
                             num_workers=1, pin_memory=True)

sample = torch.Tensor(120, 1, 28, 28).to(device)
optimizer = optim.Adam(list(pixel_cnn.parameters())+
                       list(label_net.parameters()))
criterion = torch.nn.CrossEntropyLoss()

# Training loop from jzbontar/pixelcnn-pytorch
for epoch in range(n_epochs):
    # train
    err_tr = []
    time_tr = time.time()
    pixel_cnn.train()
    label_net.train()
    for inp, lab in trainloader:
        lab = to_one_hot(lab)
        lab_emb = label_net(lab.to(device))
        inp = inp.to(device)
        target = (inp[:,0] * 255).long()
        loss = criterion(pixel_cnn(inp, lab_emb), target)
        err_tr.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    time_tr = time.time() - time_tr

    with torch.no_grad():
        # compute error on test set
        err_te = []
        time_te = time.time()
        pixel_cnn.eval()
        label_net.eval()
        for inp, lab in testloader:
            lab = to_one_hot(lab)
            lab_emb = label_net(lab.to(device))
            inp = inp.to(device)
            target = (inp[:,0] * 255).long()
            loss = criterion(pixel_cnn(inp, lab_emb), target)
            err_te.append(loss.item())
        time_te = time.time() - time_te

        # sample
        labels = torch.arange(10).repeat(12,1).flatten()
        sample.fill_(0)
        for i in range(28):
            for j in range(28):
                out = pixel_cnn(sample, label_net(to_one_hot(labels).to(device)))
                probs = F.softmax(out[:, :, i, j], dim=1)
                sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.

        utils.save_image(sample, 'sample_{:02d}.png'.format(epoch+1), nrow=10, padding=0)

        output_string = 'epoch: {}/{} bpp (train): {:.7f}' + \
            ' bpp (test): {:.7f} time (training): {:.1f}s time (testing): {:.1f}s'
        print(output_string.format(epoch+1,
                                   n_epochs,
                                   np.mean(err_tr)/np.log(2),
                                   np.mean(err_te)/np.log(2),
                                   time_tr,
                                   time_te))
