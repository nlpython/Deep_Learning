import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transforms = transforms.Compose(
	[transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 展示图片
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()




# 定义一个简单的网络类
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 定义第一层卷积神经网络, 输入通道为3, 输出通道为6, filter.size = [3, 3]
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 定义三个全连接层
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 32)
        self.fc3 = nn.Linear(32, 10)
        # 定义池化窗口
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # 在[2, 2]的池化窗口下执行最大池化操作
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc4(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        # 计算size, 除了第0个维度上的batch_size
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



net = Net()
# 定义交叉熵损失
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(20):

    running_loss = 0
    correct, total = 0, 0
    for i, data in enumerate(trainloader, 0):
        # data中包含了输入图像张量inputs, 标签张量labels
        inputs, labels = data
        # inputs.shape = [4, 3, 32, 32]
        # labels = [4, 1, 5, 7]

        # 首先将优化器梯度归零
        optimizer.zero_grad()

        # 输入图像张量进网络, 得到输出张量
        outputs = net(inputs)

        # outputs.shape = [4, 10]
        # 利用网络的输出outputs和标签labels计算损失值
        loss = criterion(outputs, labels)

        # 反向传播+梯度更新
        loss.backward()
        optimizer.step()

        # 计算accuracy
        _, predicted = torch.max(outputs, axis=1)
        correct += (predicted == labels).sum().item()
        total += predicted.size(0)
        # 打印轮次和损失值
        running_loss += loss.item()
        if (i + 1) % 500 == 0:
            print(f'epoch: {epoch + 1} | step: {i + 1} | loss: {running_loss / 500} | accuracy: {correct / total}')
            running_loss = 0

print('Finished Training.')
# # 保存模型
PATH = './model_weights/cifar_net.pth'
torch.save(net.state_dict(), PATH)
#
# dataiter = iter(testloader)
# images, labels = dataiter.next()
#
# # 展示图片
# imshow(torchvision.utils.make_grid(images))
#
# 首先实例化模型的类对象
# net = Net()
# 加载训练阶段保存好的模型的状态字典
# net.load_state_dict(torch.load(PATH))

# # 利用模型对图片进行预测
# # outputs = net(images)
# #
# # # 共有10个类别, 采用模型计算出的概率最大作为预测的类别
# # _, predicted = torch.max(outputs, 1)
# #
# # # 打印标签结果
# # print('Predictd: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs, axis=1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
# print(f'Accuracy: {correct / total}')














