import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch import nn

batch_size = 256
trans = transforms.ToTensor()

CUDA = True
CUDA = CUDA and torch.cuda.is_available()
device = torch.device("cuda:0" if CUDA else "cpu")

train_set = dataset.MNIST(root="./data", train=True, transform=trans, download=False)
test_set = dataset.MNIST(root="./data", train=False, transform=trans, download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                          batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                         batch_size=batch_size,
                                         shuffle=True)
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
      

def _BlockCNN(input, out, ks, stride=1, padd=0):
  layers = [nn.Conv2d(input, out, kernel_size=ks, padding=padd), nn.AvgPool2d(kernel_size=2, stride=2)]
  return layers
net = nn.Sequential(*_BlockCNN(1, 6, 5, padd=2), *_BlockCNN(6, 16, 5), Flatten(), nn.Linear(16*5*5, 120), nn.Sigmoid(), nn.Linear(120, 84), nn.Sigmoid(), nn.Linear(84, 10)).to(device)
def init_weights(m):
  if type(m) == nn.Linear or type(m) == nn.Conv2d:
    torch.nn.init.xavier_uniform(m.weight)
net.apply(init_weights)

criterion = nn.CrossEntropyLoss()
lr = 0.9
opt = torch.optim.SGD(net.parameters(), lr=lr)

def train_batch(X, y, opt, net, criterion):
  opt.zero_grad()
  y_hat = net(X.reshape(-1,1, 28, 28))
  loss = criterion(y_hat, y)
  loss.backward()
  opt.step()
  return loss.data

n_epochs = 20
for epoch in range(n_epochs):
  av_loss = 0
  net.train()
  for batch_idx,(X, y) in enumerate(train_loader):
    av_loss += train_batch(X.to(device), y.to(device), opt, net, criterion)
  print("epoch {}/{}, average loss : {:.5f}".format(epoch, n_epochs, av_loss))

acc = 0
for _, (X,y) in enumerate(test_loader):
  corr = torch.sum(torch.argmax(net(X.to(device)),dim=1) == y)
  acc += corr/len(X)
print("Précision sur le jeu de test : ", acc/len(test_loader))
