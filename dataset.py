from cmath import exp
import math
import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torchvision.transforms.functional as F
import random
from PIL import Image
import skfmm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from createdata import dim, max_xy, step, X, Y
from itertools import combinations
from createdata import BB
PTS = torch.from_numpy(BB).float()
pts = PTS

class ContourDataset(Dataset):
    def __init__(self, root_dir="data", split="train"):
        self.data_path = f"{root_dir}/{split}" 
        self.data = [folder for folder in os.listdir(self.data_path)]
        self.split = split

    def __len__(self):
        return len(self.data)
        #return 1

    def __getitem__(self, idx):
        folder_name = self.data[idx]
        full_path = f"{self.data_path}/{folder_name}"
        data = torch.load(f'{full_path}/data.pth')

        Pc = torch.from_numpy(data['Pc']).float()
        #Pi = torch.from_numpy(data['mesh_pts']).float()
        df = data['df']    
        df_vec = data['df_vec']
        sdf = data['sdf']  
      

        return Pc, df, sdf



def main():
    dataset = ContourDataset(split="train")
    df, sdf = dataset[0]
    print(df.shape, sdf.shape, pts.shape)

    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(x,y,z, s=30, c = df, alpha=0.1)
    #ax.scatter(Pc[:,0],Pc[:,1],Pc[:,2], c='red' )
    plt.show()
 

if __name__ == "__main__":
    main()