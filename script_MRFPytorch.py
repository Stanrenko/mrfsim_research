
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import grad
import torch.distributions as dist
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from utils_mrf import *

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class NeuralNetwork(nn.Module):
    def __init__(self, ntimesteps=175, threshold_pca=40, nparams=4):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2 * ntimesteps, threshold_pca)
        self.fc2 = nn.Linear(threshold_pca, 200)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc3 = nn.Linear(200, 25)
        self.dropout3 = nn.Dropout2d(0.5)
        self.fc4 = nn.Linear(25, nparams)

    # x represents our data
    def forward(self, x):
        x = self.fc1(x)
        # x = F.linear(x)

        # x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)

        # x = self.dropout3(x)
        x = self.fc3(x)
        out = F.relu(x)

        x = self.fc4(x)
        out = F.relu(x)
        return out


input_scaler=StandardScaler()
output_scaler=MinMaxScaler()


keys,values=read_mrf_dict("./mrf175_SimReco2_light.dict",FF_list=np.arange(0,1,0.05))

values=values/np.expand_dims(np.linalg.norm(values,axis=-1),axis=1)

j=np.random.choice(range(values.shape[0]))

plt.figure()
plt.plot(np.angle(values[j,:]))

pca=PCAComplex()
pca.fit(values)
plt.figure()
plt.plot(np.angle(pca.components_[:,0]))

num_scales=5
phi_s=np.arange(-np.pi,np.pi+0.1,2*np.pi/num_scales)
lambda_s=np.array(1)

values_all=np.expand_dims(values,axis=(0,1))
lambda_s=np.expand_dims(lambda_s,axis=0)
phi_s=np.expand_dims(phi_s,axis=1)
lambda_s,phi_s=np.broadcast_arrays(lambda_s,phi_s)

phi_s=np.expand_dims(phi_s,axis=(2,3))
lambda_s=np.expand_dims(lambda_s,axis=(2,3))

values_all_scaled=lambda_s*np.exp(1j*phi_s)*values_all

keys_all=np.expand_dims(np.array(keys),axis=(0,1))
keys_all=np.tile(keys_all,(num_scales+1,num_scales,1,1))

keys_all=np.reshape(keys_all,(-1,keys_all.shape[-1]))
values_all_scaled=np.reshape(values_all_scaled,(-1,values_all_scaled.shape[-1]))


Y_TF = np.array(np.array(keys_all)[:,[0,2,3,4]])

#signal=signal/np.expand_dims(np.linalg.norm(signal,axis=-1),axis=-1)
real_signal = values_all_scaled.real
imag_signal = values_all_scaled.imag

#real_signal=real_signal/np.expand_dims(np.linalg.norm(real_signal,axis=-1),axis=-1)
#imag_signal=imag_signal/np.expand_dims(np.linalg.norm(imag_signal,axis=-1),axis=-1)

X_TF = np.concatenate((real_signal, imag_signal), axis=1)
#X_TF = X_TF/np.expand_dims(np.linalg.norm(X_TF,axis=-1),axis=-1)

X_TF = input_scaler.fit_transform(np.concatenate((real_signal, imag_signal), axis=1))
Y_TF_norm=output_scaler.fit_transform(Y_TF)

batch_size = 1024
epochs = 1000
lr = 0.1

X = torch.Tensor(X_TF) # transform to torch tensor
Y = torch.Tensor(Y_TF_norm)

my_dataset = TensorDataset(X,Y) # create your datset
my_dataloader = DataLoader(my_dataset,batch_size=batch_size,shuffle=True)

train_ds, test_ds = torch.utils.data.random_split(my_dataloader.dataset, [int(0.9*values_all_scaled.shape[0]), values_all_scaled.shape[0]-int(0.9*values_all_scaled.shape[0])])
train_dataloader=DataLoader(train_ds,batch_size=batch_size,shuffle=True)
test_dataloader=DataLoader(test_ds,batch_size=batch_size,shuffle=True)

model=NeuralNetwork()
loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer=torch.optim.Adam(model.parameters())
lmbda = lambda epoch: 0.001*0.69 ** (int(epoch/10))
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
scheduler=None


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    running_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.shape[0]

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return running_loss / size


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    running_loss = 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_fn(pred, y)
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            running_loss += loss.item() * X.shape[0]

    running_loss /= size
    # correct /= size
    print("Test Error: Avg loss: {}".format(running_loss))
    return running_loss


loss_values = []
test_loss_values = []
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    model.train()
    running_loss = train_loop(train_dataloader, model, loss, optimizer)
    if scheduler is not None:
        scheduler.step()
    loss_values.append(running_loss)

    model.eval()
    test_loss = test_loop(test_dataloader, model, loss)
    test_loss_values.append(test_loss)
    # test_loop(test_dataloader, model, loss_fn)
print("Done!")

plt.figure()
plt.plot(loss_values[200:])
plt.plot(test_loss_values[200:])


i =np.random.choice(X.shape[0])
with torch.no_grad():
    plt.figure()
    Y_predict=model((X[i,:].reshape(1,-1))).numpy()
    plt.plot(Y_predict.squeeze(),label="Estimated")
    plt.plot((Y[i,:]).numpy(),label="Orig")
    print("Orig params {} Found params {}".format(output_scaler.inverse_transform(Y[i,:].reshape(1,-1)),output_scaler.inverse_transform(Y_predict)))
    plt.legend()