import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from logger import Logger

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        # Difference between nn.funcional vs. nn -> https://discuss.pytorch.org/t/whats-the-difference-between-torch-nn-functional-and-torch-nn/681
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        # What is view? -> https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)

def train(args, model, device, train_loader, optimizer, epoch, logger):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data means local data, target means local labels
        # more about data loader : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel.html

        data, target = data.to(device), target.to(device)
        # What is .to() function? It is for assigning data to different devices.
        # For more detail -> https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html#sphx-glr-beginner-blitz-data-parallel-tutorial-py
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        avg_loss = 0
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
            avg_loss += loss.item()

        avg_loss /= len(train_loader)
        info = {'loss': avg_loss}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)

def test(args, model, device, test_loader, logger):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        # What is no_grad? Check out 'autograd' -> https://pytorch.org/tutorials/beginner/former_torchies/autograd_tutorial.html
        # Also see torch.Tensor doc about methods of data, target in the below. -> https://pytorch.org/docs/stable/tensors.html#torch.Tensor
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
        
def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',\
            help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',\
            help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',\
            help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',\
            help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',\
            help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,\
            help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',\
            help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',\
            help='how many batches to wait before logging training status')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # About pinned memory -> https://www.cs.virginia.edu/~mwb7w/cuda_support/pinned_tradeoff.html

    train_loader = torch.utils.data.DataLoader(\
            datasets.MNIST('../data', train=True, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),\
            batch_size = args.batch_size, shuffle = True, **kwargs)

    test_loader = torch.utils.data.DataLoader(\
            datasets.MNIST('../data', train=False, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),\
            batch_size = args.test_batch_size, shuffle = True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum = args.momentum)
    logger = Logger('./log')

    for epoch in range(1, args.epochs +1):
        train(args, model, device, train_loader, optimizer, epoch, logger)
        test(args, model, device, test_loader, logger)

if __name__ == '__main__':
    main()

    

    


