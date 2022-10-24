import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tarfile
import numpy as np
from time import time
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision.transforms as tt
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import SVHN
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from copy import deepcopy
import pickle


########################################################################

#Source : https://jovian.ai/aakashns/05b-cifar10-resnet

def show_example(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img)

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

@torch.no_grad()
def evaluate(model, val_loader, dropout, rate, cell_list=None, block_list=None):
    model.train()
    outputs = [model.validation_step(batch, dropout, rate, cell_list, block_list) for batch in val_loader]
    return model.validation_epoch_end(outputs)
"""""
@torch.no_grad()
def evaluate(model, val_loader):
    model.train()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
"""


class ImageClassificationBase(nn.Module):

    def training_step_ghost(self, batch, dropout, rate, ghost) :
        i = ghost
        out_total = []
        while i <= len(batch[0]) :
            ghost_batch = [batch[0][i-ghost:i], batch[1][i-ghost:i]]
            images, labels = ghost_batch
            out = self(images, dropout, rate)  #None, None, True
            out_total.append(out)
            i += ghost
        if i > len(batch[0]) and i-ghost != len(batch[0]) :
            ghost_batch = [batch[0][i-ghost:len(batch[0])], batch[1][i-ghost:len(batch[0])]]
            images, labels = ghost_batch
            out = self(images, dropout, rate)
            out_total.append(out)
        out_total = torch.cat(tuple([*out_total]), dim=0)
        images, labels = batch
        loss = F.cross_entropy(out_total, labels)
        return loss

    def training_step(self, batch, dropout, rate):
        images, labels = batch 
        out = self(images, dropout, rate)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch, dropout, rate, cell_list, block_list):
        images, labels = batch 
        out = self(images)      #[[[1,2,3,4,5,6,7] for i in range(4)] for j in range(3)]    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
    """""
    def validation_step(self, batch):
        images, labels = batch
        out = self(images, 0, 0, None, None, False)          # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
    """
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

####################################################################################


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr, momentum=0.9, nesterov=True)
    dropout = 0.06 #0.08
    rate = 0
    epoch = 0
    stop = True
    for epoch in range(epochs) :
        # Training Phase 
        print("epoch : ", epoch+1)
        start = time()
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step_ghost(batch, dropout, rate, 32)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        rate += 1/epochs
        result = evaluate(model, val_loader, 0, 0)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        end = time()
        print(end-start)
    return history


def ConvBlock(in_channels, out_channels, k_size) :
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding='same'),
    )

class ChoiceBlock(nn.Module) :
    def __init__(self, in_channels, out_channels) :         #16 and 32
        super().__init__()
        self.l2_1 = nn.Sequential(
            ConvBlock(in_channels, 8, 1),
            ConvBlock(8, 16, 3),
            ConvBlock(16, out_channels, 3)
        )
        self.l2_2 = nn.Sequential(
            ConvBlock(in_channels, 8, 1),
            ConvBlock(8, 16, 5),
            ConvBlock(16, out_channels, 5)
        )
        self.l2_3 = nn.Sequential(
            ConvBlock(in_channels, 8, 1),
            ConvBlock(8, 16, 7),
            ConvBlock(16, out_channels, 7)
        )
        self.l2_4 = nn.Sequential(
            ConvBlock(in_channels, 8, 1),
            ConvBlock(8, 16, (1, 7)),
            ConvBlock(16, out_channels, (7, 1))
        )
        self.l2_5 = nn.Sequential(
            ConvBlock(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1, padding=1)
        )
        self.l2_6 = nn.Sequential(
            ConvBlock(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AvgPool2d(3, stride=1, padding=1)
        )
        self.l2_7 = nn.Sequential(
            ConvBlock(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Identity()
        )

    def choice(self, xb, id) :
        if id == 1 :
            return self.l2_1(xb)
        elif id == 2 :
            return self.l2_2(xb)
        elif id == 3 :
            return self.l2_3(xb)
        elif id == 4 :
            return self.l2_4(xb)
        elif id == 5 :
            return self.l2_5(xb)
        elif id == 6 :
            return self.l2_6(xb)
        elif id == 7 :
            return self.l2_7(xb)
        else :
            print(id)
            print("Wrong ID")
        

    def forward(self, xb, id) :
        
        out = self.choice(xb, id[0])
        for i in range(1, len(id)) :
            if id[i] != 0 :
                out += self.choice(xb, id[i])
        return out
        """
        out1 = self.choice(xb, id[0])
        if id[1] == 0 : return out1
        else : 
            out2 = self.choice(xb, id[1])  
            
            if len(id) > 2 :
                print(id)
                for i in range(2, len(id)) :
                    if id[i] != 0 :
                        out2 += self.choice(xb, id[i])
        """
            #return out1 + out2


def avPool(not_first) :
        if not_first : return nn.AvgPool2d(4)
        else : return nn.AvgPool2d(2)


class Cell(nn.Module) :
    def __init__(self, in_channelsA, in_channelsB, out_channels, not_first=True) :
        super().__init__()
        self.not_first = not_first
        self.l1A = ConvBlock(in_channelsA, out_channels, 1)
        self.l1B = ConvBlock(in_channelsB, out_channels, 1)
        self.avg1 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.avg2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            avPool(not_first)
        )
        self.choice1 = ChoiceBlock(out_channels*2, out_channels)
        self.choice2 = ChoiceBlock(out_channels*3, out_channels)
        self.choice3 = ChoiceBlock(out_channels*4, out_channels)
        self.choice4 = ChoiceBlock(out_channels*5, out_channels)



    def concat(self, to_concat, cell_list, block, id) :
        out = []
        x = 0
        for i in cell_list :
            if i == 1 : out.append(to_concat[0])
            elif i == 2 : out.append(to_concat[1])
            elif i == 3 : out.append(to_concat[2])
            elif i == 4 : out.append(to_concat[3])
            elif i == 5 : out.append(to_concat[4])
            x += 1
        pad = id + 1 - x
        for i in range(pad) :
            out.append(torch.zeros(to_concat[0].size()).to(get_default_device()))
        out = torch.cat(tuple([*out]), dim=1)
        if id == 1 : out = self.choice1(out, block)
        elif id == 2 : out = self.choice2(out, block)
        elif id == 3 : out = self.choice3(out, block)
        else : out = self.choice4(out, block)
        return out


    def forward(self, xbA, xbB, cell_list, block_list, flag=True) :
        outA = self.l1A(xbA)
        outB = self.l1B(xbB)
        if flag :
            outA = self.avg1(outA)
            outB = self.avg2(outB)
        elif self.not_first : outB = self.avg1(outB)
        try :
            out1 = self.concat([outA, outB], cell_list[0], block_list[0], 1)
            out2 = self.concat([outA, outB, out1], cell_list[1], block_list[1], 2)
            out3 = self.concat([outA, outB, out1, out2], cell_list[2], block_list[2], 3)
            out4 = self.concat([outA, outB, out1, out2, out3], cell_list[3], block_list[3], 4)
        except :
            print(cell_list)
            print()
            print(block_list)
        return torch.cat((outA, outB, out1, out2, out3, out4), dim=1)


class SuperNetwork(ImageClassificationBase) :
    def __init__(self, in_channels) :
        super().__init__()
        self.stem1 = ConvBlock(in_channels, 8, 3)
        self.stem2 = ConvBlock(in_channels, 8, 3)
        self.cell1 = Cell(8, 8, 16, False)
        self.cell2 = Cell(16*6, 8, 32)
        self.cell3 = Cell(32*6, 16*6, 64, False)
        self.endL = ConvBlock(64*6, 10, 1)  #64
        self.avg = nn.Sequential( 
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.AvgPool2d(8, 8),   #(2, 2)
        )
        self.flatten = nn.Flatten()


    def forward(self, xb, dropout, rate, cell_list=None, block_list=None, train=False) :
        if cell_list == None :
            tmp_cell_list = [[], [], [], []]
            for i in range(4) :
                for j in range(i+2) :
                    if np.random.random() > rate*(dropout)**(1/(i+2)) :
                        tmp_cell_list[i].append(j+1)
            cell_list = [tmp_cell_list, tmp_cell_list, tmp_cell_list]

        if block_list == None :
            tmp_block_list = [[], [], [], []]
            for i in range(4) :
                
                for j in range(1, 8) :
                    if np.random.random() > rate*(dropout)**(1/4) :
                        tmp_block_list[i].append(j)
                if tmp_block_list[i] == [] :
                    tmp_block_list[i].append(np.random.randint(1, 8))
            block_list = [tmp_block_list, tmp_block_list, tmp_block_list]

        out1 = self.stem1(xb)
        out2 = self.stem2(xb)
        out_cell1 = self.cell1(out2, out1, cell_list[0], block_list[0])
        out_cell2 = self.cell2(out_cell1, out2, cell_list[1], block_list[1], False)
        out_cell3 = self.cell3(out_cell2, out_cell1, cell_list[2], block_list[2])
        out = self.endL(out_cell3)
        out = self.avg(out)
        return self.flatten(out)


def createFC(input, output, hidden=128) :
    fc = nn.Sequential(
        nn.ReLU(),
        nn.Linear(input, hidden),
        nn.ReLU(),
        nn.Linear(hidden, output),
    )
    return fc


class Architect(nn.Module) :
    def __init__(self, model) :
        super().__init__()
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.get_arch_parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=0.001)
        self.out_total = []

    def _backward_step(self, batch, dropout, rate) :
        loss = self.model.training_step(batch, dropout, rate)
        loss.backward()

    def step(self, batch, dropout, rate) :
        self.optimizer.zero_grad()
        self._backward_step(batch, dropout, rate)
        self.optimizer.step()
        self.model.update()

    def train(self, val_loader, dropout, rate, it=1) :
        self.optimizer.zero_grad()
        out_total = []
        labels = []
        for i in range(it) :
            inputs, classes = next(iter(val_loader))
            out = self.model(inputs, dropout, rate)
            out_total.append(out)
            labels.append(classes)
        out_total = torch.cat(tuple([*out_total]), dim=0)
        labels = torch.cat(tuple([*labels]), dim=0)
        loss = F.cross_entropy(out_total, labels)
        loss.backward()
        self.optimizer.step()
        self.model.update()


class DartChoiceBlock(nn.Module) :
    def __init__(self, in_channels, out_channels, alpha) :
        super().__init__()
        self.alpha = alpha
        self.softmax = nn.Softmax(-1)
        self.l2_1 = nn.Sequential(
            ConvBlock(in_channels, 16, 1),
            ConvBlock(16, 32, 3),
            ConvBlock(32, out_channels, 3)
        )
        self.l2_2 = nn.Sequential(
            ConvBlock(in_channels, 16, 1),
            ConvBlock(16, 32, 5),
            ConvBlock(32, out_channels, 5)
        )
        self.l2_3 = nn.Sequential(
            ConvBlock(in_channels, 16, 1),
            ConvBlock(16, 32, 7),
            ConvBlock(32, out_channels, 7)
        )
        self.l2_4 = nn.Sequential(
            ConvBlock(in_channels, 16, 1),
            ConvBlock(16, 32, (1, 7)),
            ConvBlock(32, out_channels, (7, 1))
        )
        self.l2_5 = nn.Sequential(
            ConvBlock(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1, padding=1)
        )
        self.l2_6 = nn.Sequential(
            ConvBlock(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AvgPool2d(3, stride=1, padding=1)
        )
        self.l2_7 = nn.Sequential(
            ConvBlock(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Identity()
        )

    def update(self, alpha) :
        self.alpha = alpha

    def choice(self, xb, alpha) :
        prob = self.softmax(alpha)
        out = self.l2_1(xb)*prob[0] + self.l2_2(xb)*prob[1] + self.l2_3(xb)*prob[2] + self.l2_4(xb)*prob[3] \
            + self.l2_5(xb)*prob[4] + self.l2_6(xb)*prob[5] + self.l2_7(xb)*prob[6]
        return out
        

    def forward(self, xb) :
        out1 = self.choice(xb, self.alpha)
        return out1

    def get_new_alpha(self, alpha) :
        self.alpha = alpha


class DartCell(nn.Module) :
    def __init__(self, in_channelsA, in_channelsB, out_channels, alpha, not_first=True) :
        super().__init__()
        self.not_first = not_first
        self.l1A = ConvBlock(in_channelsA, out_channels, 1)
        self.l1B = ConvBlock(in_channelsB, out_channels, 1)
        self.avg1 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.choice1 = DartChoiceBlock(out_channels*2, out_channels, alpha[0])
        self.choice2 = DartChoiceBlock(out_channels*3, out_channels, alpha[1])
        self.choice3 = DartChoiceBlock(out_channels*4, out_channels, alpha[2])
        self.choice4 = DartChoiceBlock(out_channels*5, out_channels, alpha[3])


    def update(self, alpha) :
        self.choice1.update(alpha[0])
        self.choice2.update(alpha[1])
        self.choice3.update(alpha[2])
        self.choice4.update(alpha[3])

    def concat(self, to_concat, cell_list, id) :
        out = []
        x = 0
        for i in cell_list :
            if i == 1 : out.append(to_concat[0])
            elif i == 2 : out.append(to_concat[1])
            elif i == 3 : out.append(to_concat[2])
            elif i == 4 : out.append(to_concat[3])
            elif i == 5 : out.append(to_concat[4])
            x += 1
        pad = id + 1 - x
        for i in range(pad) :
            out.append(torch.zeros(to_concat[0].size()).to(get_default_device()))
        out = torch.cat(tuple([*out]), dim=1)
        if id == 1 : out = self.choice1(out)
        elif id == 2 : out = self.choice2(out)
        elif id == 3 : out = self.choice3(out)
        else : out = self.choice4(out)
        return out


    def forward(self, xbA, xbB, cell_list, flag=True) :
        outA = self.l1A(xbA)
        outB = self.l1B(xbB)
        if flag :
            outA = self.avg1(outA)
            outB = self.avg1(outB)
        if self.not_first : outB = self.avg1(outB)
        out1 = self.concat([outA, outB], cell_list[0], 1)
        out2 = self.concat([outA, outB, out1], cell_list[1], 2)
        out3 = self.concat([outA, outB, out1, out2], cell_list[2], 3)
        out4 = self.concat([outA, outB, out1, out2, out3], cell_list[3], 4)
        return torch.cat((outA, outB, out1, out2, out3, out4), dim=1)


class DartNetwork(ImageClassificationBase) :
    def __init__(self, in_channels) :
        super().__init__()
        self.stem1 = ConvBlock(in_channels, 8, 1)
        self.stem2 = ConvBlock(in_channels, 8, 1)
        self.arch_parameters = torch.nn.Parameter(data=torch.zeros(12, 7).to(get_default_device()), requires_grad=True)
        self.cell1 = DartCell(8, 8, 16, self.arch_parameters[0:4], False)
        self.cell2 = DartCell(16*6, 8, 32, self.arch_parameters[4:8])
        self.cell3 = DartCell(32*6, 16*6, 64, self.arch_parameters[8:12], False)
        self.endL = ConvBlock(64*6, 10, 1)
        self.avg = nn.Sequential( 
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.AvgPool2d(8, 8)
        )
        self.flatten = nn.Flatten()

    def get_arch_parameters(self) :
        return [self.arch_parameters]

    def get_final_arch(self) :
        indexes = []
        for i in self.arch_parameters :
            print(i)
            ind = [index for index, item in enumerate(i) if item == max(i)]
            indexes.append(ind)
        return indexes


    def update(self) :
        self.cell1.update(self.arch_parameters[0:4])
        self.cell2.update(self.arch_parameters[4:8])
        self.cell3.update(self.arch_parameters[8:12])


    def forward(self, xb, dropout, rate, a=None, b=None) :
        cell_list = [[], [], [], []]
        for i in range(4) :
            for j in range(i+2) :
                if np.random.random() > rate*(dropout)**(1/(i+2)) :
                    cell_list[i].append(j+1)

        out1 = self.stem1(xb)
        out2 = self.stem2(xb)
        out_cell1 = self.cell1(out2, out1, cell_list)
        out_cell2 = self.cell2(out_cell1, out2, cell_list, False)
        out_cell3 = self.cell3(out_cell2, out_cell1, cell_list)
        out = self.endL(out_cell3)
        out = self.avg(out)
        return self.flatten(out)

        
def Evolutionist(model, val_loader, P=50, C=2000, S=5, old=True, param=True) :
    population = []
    tot = P
    best = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
    hist = []
    while len(population) < P :
        arch = RandomArch()
        arch_val = evalu(model, val_loader, arch, param)
        population.append([arch, arch_val])
        hist.append(arch_val)
        for j in range(len(best)):
            if arch_val > best[j][0]:
                best.insert(j, [arch_val, arch])
                del best[10]
                break
    while tot < C :
        echantillon = []
        while len(echantillon) < S :
            num = np.random.randint(0, P)
            if population[num] not in echantillon :
                echantillon.append(population[num])
        max = 0
        parent = None
        for i in range(len(echantillon)) :
            if echantillon[i][1] > max :
                max = echantillon[i][1]
                parent = echantillon[i][0]
        enfant = mutation2(parent)
        arch_val = evalu(model, val_loader, enfant, param)
        hist.append(arch_val)
        if old : del population[0]
        else :
            min = 100
            drop = None
            for i in range(len(population)) :
                if population[i][1] < min :
                    min = population[i][1]
                    drop = i
            del population[drop]
        population.append([enfant, arch_val])
        tot += 1

        for j in range(len(best)):
            if arch_val > best[j][0]:
                best.insert(j, [arch_val, enfant])
                del best[10]
                break
    return best


def mutation2(arch) :
    new_arch = ""
    index1 = np.random.randint(3)
    index2 = np.random.randint(14)
    for i in range(len(arch)) :
        if i == (index2 + 42*index1) :
            if arch[i] == '1' :
                new_arch += '0'
            else : new_arch += '1'
        else : new_arch += arch[i]
    return new_arch


def mutation(arch) :
    index = np.random.randint(0, len(arch))
    index2 = 999
    for k in [14, 21, 28, 35, 56, 63, 70, 77, 98, 105, 112, 119] :
        if (k <= index) and (index <= k+6) :
            tot = 0
            for i in range(7) :
                if arch[k+i] == '1' :
                    tot += 1
            if (arch[index] == '1' and tot == 1) or (arch[index] == '0' and tot == 2) :
                index2 = np.random.randint(k, k+7)
                while index2 == index or (arch[index] == '0' and tot == 2 and arch[index2] == '0') :
                    index2 = np.random.randint(k, k + 7)
            break
    new_arch = ""
    for i in range(len(arch)) :
        if i == index or i == index2 :
            if arch[i] == '1' :
                new_arch += '0'
            else : new_arch += '1'
        else : new_arch += arch[i]
    return new_arch


def evalu(model, val_loader, arch, p) :
    cell_list, block_list = translate_arch(arch)
    res = evaluate(model, val_loader, 0, 0, cell_list, block_list)
    if not p : return res['val_acc']
    param = number_parameters(block_list, cell_list)
    arch_val = eval_arch(param, res['val_acc'])
    return [arch_val, res['val_acc']]


def translate_arch(arch) :
    cell_list = []
    block_list = []
    for i in range(3) :
        subcell = []
        subblock = []
        subsubcell = []
        for j in range(14) :
            if arch[(i*42+j)] == '1' :
                l = [1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5]
                subsubcell.append(l[j])
            if j == 1 or j == 4 or j == 8 :
                subcell.append(subsubcell)
                subsubcell = []
        subcell.append(subsubcell)
        for j in range(4) :
            subsubblock = []
            for k in range(7) :
                if arch[(i*42+14+j*7+k)] == '1' :
                    subsubblock.append(k+1)
            if len(subsubblock) == 1 :
                subsubblock.append(0)
            subblock.append(subsubblock)
        block_list.append(subblock)
        cell_list.append(subcell)
    return cell_list, block_list


def RandomArch() :
    arch = ""
    for i in range(3) :
        for j in range(14) :
            num = np.random.randint(2)
            arch += str(num)
        for k in range(4) :
            num = np.random.randint(2)
            choice1 = np.random.randint(0, 7)
            if num :
                choice2 = np.random.randint(0, 7)
                while choice2 == choice1 :
                    choice2 = np.random.randint(0, 7)
                for j in range(7) :
                    if j == choice1 or j == choice2 :
                        arch += '1'
                    else : arch += '0'
            else :
                for j in range(7) :
                    if j == choice1 :
                        arch += '1'
                    else : arch += '0'
    return arch


def RandomAlgo() :

    cell_list = []
    for k in range(3) :
        tmp_cell_list = [[], [], [], []]
        for i in range(4) :
            for j in range(i+2) :
                if np.random.randint(2) :
                    tmp_cell_list[i].append(j+1)
        cell_list.append(tmp_cell_list)
        

    block_list = []
    for k in range(3) :
        tmp_block_list = [[], [], [], []]
        for i in range(4) :
            num = np.random.randint(1, 8)
            tmp_block_list[i].append(num)
            if np.random.randint(2) :
                num2 = np.random.randint(1, 8)
                while num2 == num :
                    num2 = np.random.randint(1, 8)
                tmp_block_list[i].append(num2)
            #else : tmp_block_list[i].append(0)
        block_list.append(tmp_block_list)

    return cell_list, block_list


def RandomSearch(model, val_loader, it=5000) :
    best1 = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
    best2 = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
    rand = []
    hist = []
    flag = True
    for i in range(it) :
        cell_list, block_list = RandomAlgo()
        res = evaluate(model, val_loader, 0, 0, cell_list, block_list)
        hist.append(res['val_acc'])
        param = number_parameters(block_list, cell_list)
        arch_val = eval_arch(param, res['val_acc'])
        for j in range(len(best1)) :
            if arch_val > best1[j][0] :
                best1.insert(j, [arch_val, res['val_acc'], param, cell_list, block_list])
                del best1[10]
                break
        for j in range(len(best2)):
            if res['val_acc'] > best2[j][0] :
                best2.insert(j, [res['val_acc'], param, cell_list, block_list])
                del best2[10]
                break
        if flag and best1[9][0] != 0 :
            rand = deepcopy(best1)
            flag = False
    return best1, best2, rand

def dartfit(epochs, lr, model, train_loader, val_loader, test_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr, momentum=0.9, nesterov=True)
    dropout = 0.06
    rate = 0
    arch = Architect(model)
    arch.to(get_default_device())
    for epoch in range(epochs):
        # Training Phase
        print("epoch : ", epoch + 1)
        print(model.get_arch_parameters()[0][0])
        model.train()
        train_losses = []
        for batch in train_loader :
            arch.train(val_loader, dropout, rate)
            loss = training_step_ghostdart(model, batch, dropout, rate, 32)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        rate += 1 / epochs
        result = evaluate(model, test_loader, 0, 0, None, None)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def training_step_ghostdart(self, batch, dropout, rate, ghost) :
    i = ghost
    out_total = []
    while i <= len(batch[0]) :
        ghost_batch = [batch[0][i-ghost:i], batch[1][i-ghost:i]]
        images, labels = ghost_batch
        out = self(images, dropout, rate)
        out_total.append(out)
        i += ghost
    if i > len(batch[0]) and i-ghost != len(batch[0]) :
        ghost_batch = [batch[0][i-ghost:len(batch[0])], batch[1][i-ghost:len(batch[0])]]
        images, labels = ghost_batch
        out = self(images, dropout, rate)
        out_total.append(out)
    out_total = torch.cat(tuple([*out_total]), dim=0)
    images, labels = batch
    loss = F.cross_entropy(out_total, labels)
    return loss


def number_parameters(block_list, cell_list) :
    total = 100610
    for i in range(len(block_list)) :
        filters = 2*(2**(i))*16
        for j in range(4) :
            for k in range(len(block_list[i][j])) :
                if cell_list[i][j] != [] :
                    if block_list[i][j][k] == 1 :
                        total += len(cell_list[i][j])*filters*16 + 48 + 9*16*32 + 9*32*filters + filters + 96 + 2*len(cell_list[i][j])*filters
                    elif block_list[i][j][k] == 2 :
                        total += len(cell_list[i][j])*filters*16 + 48 + 25*16*32 + 25*32*filters + filters + 96 + 2*len(cell_list[i][j])*filters
                    elif block_list[i][j][k] == 3 :
                        total += len(cell_list[i][j])*filters*16 + 48 + 49*16*32 + 49*32*filters + filters + 96 + 2*len(cell_list[i][j])*filters
                    elif block_list[i][j][k] == 4 :
                        total += len(cell_list[i][j])*filters*16 + 48 + 7*16*32 + 7*32*filters + filters + 96 + 2*len(cell_list[i][j])*filters
                    elif block_list[i][j][k] != 0 :
                        total += 2*filters + len(cell_list[i][j])*filters*filters + filters  + 2*len(cell_list[i][j])*filters
    return total


def eval_arch(param, acc, alpha=1, beta=1) :

    Pval = 1 - param/2232634       
    if Pval < 0.1 : Pval = 0.1
    return (alpha*np.log(10*Pval) + beta*10*acc)

def reseval(param, score) :
    return score - np.log(10*(1 - param/2232634))

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params