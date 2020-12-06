import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch import nn

batch_size = 256
trans = transforms.ToTensor()

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
       
 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv2d1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2d2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = Flatten()
        self.lin1 = nn.Linear(16*5*5, 120)
        self.lin2 = nn.Linear(120, 84)
        self.lin3 = nn.Linear(84, 10)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, X):
        x = self.sigmoid(self.conv2d1(X.reshape(-1,1, 28, 28)))
        x = self.avgpool1(x)
        x = self.sigmoid(self.conv2d2(x))
        x = self.avgpool2(x)
        x = self.flatten(x)
        x = self.sigmoid(self.lin1(x))
        x = self.sigmoid(self.lin2(x))
        x = self.lin3(x)
        
        return x

net = Net()
def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform(m.weight)
net.apply(init_weights)

criterion = nn.CrossEntropyLoss()
lr = 0.9
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

for epoch in range(10):
    av_loss = 0
    for batch_idx, (X,y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_hat = net(X)
        with torch.enable_grad():
            loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        av_loss += loss.data
        if (batch_idx+1) == len(train_loader):
            print("epoch {}, loss : {:.5f}".format(epoch+1, av_loss/len(train_loader)))


acc = 0
for _,(X,y) in enumerate(test_loader):
    corr = 0
    for idx in range(len(X)):
        if torch.argmax(net(X[idx])) == y[idx]:
            corr += 1
    acc += corr/len(X)
print("Précision sur le jeu de test : ", acc/len(test_loader))
