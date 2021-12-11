import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# 导入数据
train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=trans)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=trans)

# 每64个进行一次打包
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

writer = SummaryWriter('../dataloader')
step = 0
for data in test_loader:
    imgs, targets = data
    # print(imgs.shape)
    # print(targets)
    writer.add_images('data', imgs, step)
    step += 1

writer.close()

