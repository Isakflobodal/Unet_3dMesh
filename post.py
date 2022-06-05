import torch
from NN import NeuralNet
from dataset import ContourDataset 
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
import torchvision.models as models

from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
#from skimage import data, img_as_float
from createdata import BB, dim, step
import numpy as np
from scipy.signal import argrelextrema

def FindPeaks(pred):
    pred = pred.flatten()
    coords = []
    for index, dist in enumerate(pred):
        if dist < step:
            coord = pts[index]
            coords.append(coord)
    return np.array(coords)

import numpy as np
import scipy.ndimage as ndimage

def Peaks1(pred, pts, order=10):
    pred *= -1
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0
    
    filtered = ndi.maximum_filter(pred, footprint=footprint)
    mask_local_maxima = pred > filtered
    coords = np.asarray(np.where(mask_local_maxima)).T
    coords_retur = pts[mask_local_maxima]
    return coords_retur

def Peaks(pred, pts, order=1):
    pred *= -1
    peaks0 = np.array(argrelextrema(pred, np.greater, axis=0, order=order))
    peaks1 = np.array(argrelextrema(pred, np.greater, axis=1, order=order))
    peaks2 = np.array(argrelextrema(pred, np.greater, axis=2, order=order))

    stacked = np.vstack((peaks0.transpose(), peaks1.transpose(), peaks2.transpose()))

    elements, counts = np.unique(stacked, axis=0, return_counts=True)
    coords = elements[np.where(counts == 3)[0]]
    coords_retur = pts[coords[:, 0], coords[:, 1], coords[:, 2]]

    return coords_retur




def PlotCoords(pred, pts, coords_pred):
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    max, min = x.max().item(), x.min().item()
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # ax = fig.add_subplot(1,2,1, projection = '3d')
    # a = ax.scatter(x,y,z, s=30, c = target, cmap = 'hsv', alpha=0.1)
    # ax.set_title('Target')
    # ax.set_xlim(min,max)
    # ax.set_ylim(min,max)
    # ax.set_zlim(min,max)

    ax = fig.add_subplot(1,1,1, projection = '3d')
    ax.scatter(x,y,z, s=30, c = pred, cmap = 'RdBu', alpha=0.005)
    #ax.scatter(coords_target[:,0],coords_target[:,1],coords_target[:,2], c='green' )
    ax.scatter(coords_pred[:,0],coords_pred[:,1],coords_pred[:,2], c='black' )
    #ax.set_title('Prediction')
    ax.set_xlim(min,max)
    ax.set_ylim(min,max)
    ax.set_zlim(min,max)
    plt.axis('off')
    plt.show()


# load trained model
config = torch.load("model_check_point.pth")
model = NeuralNet(device="cpu")
model.load_state_dict(config["state_dict"])
model.eval()

# get dummy data 
test_data = ContourDataset(split="test")
test_loader = DataLoader(test_data, batch_size=1)
it = iter(test_loader)
Data = next(it)
Data = next(it)



Pc, df, sdf = Data


df_target = df.squeeze().detach().cpu().numpy()     # (40,40,40)
df_pred = model(sdf)                                # [1,40,40,40]
#df_pred = torch.linalg.norm(df_pred, dim=-1)        # 

df_pred = df_pred.detach().cpu().numpy().squeeze()  # (40,40,40)
print(df_target.shape)
print(df_pred.shape)

pts = BB                                            # (6400,3)
pts_ = pts.reshape(dim,dim,dim,3)                    # (40,40,40,3)

#coords_target = Peaks1(df_target, pts_)
coords_pred = Peaks1(df_pred, pts_)

PlotCoords(df_pred, pts, coords_pred)
