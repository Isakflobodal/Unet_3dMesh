import numpy as np
import math
import matplotlib.pyplot as plt
import random
import pygmsh
import os
import torch
from itertools import combinations
from PIL import Image, ImageDraw
import skfmm
import torchvision.transforms.functional as F
import time
import gmsh

from statistics import pvariance
import numpy as np
import pyvista as pv
import meshio 
import matplotlib as plt
import os
import random as rnd


# gets a random float between 0.2 and 1.0, that are weighted towards larger numbers
def WeightedRandomRadius():
    list_random = [random.uniform(0.2,0.3), random.uniform(0.3,0.4), random.uniform(0.4,0.6),random.uniform(0.6,0.8),random.uniform(0.8,1.0)]
    weights = (5,5,25,30,35)
    return random.choices(list_random, weights, k=1)[0]

# create a random contour with a given number of edges
def GetRandomContour(nEdges):
    contour = []
    one_rad = np.pi/180
    rad = (2*np.pi)/nEdges
    theta = 0
    for i in range(nEdges):
        theta_ = random.uniform(theta+one_rad*10, theta + rad-one_rad*10)
        radius_ = WeightedRandomRadius()
        contour.append([radius_*np.cos(theta_),radius_*np.sin(theta_)])
        theta += rad
    return contour

def GetSignedDistance(contour: np.array):
    # create image where inside is white and outside black
    contour = (contour+max_xy)/(max_xy*2)
    a = (contour*dim).reshape(-1).tolist()
    img = Image.new("RGB", size=[dim, dim], color="black")
    img1 = ImageDraw.Draw(img)
    img1.polygon(a, outline ="white", fill="white")

    # convert image to tensor
    ima = np.array(img)

    # differentiate the inside / outside rigion
    phi = np.int64(np.any(ima[:,:,:3], axis = 2))
    phi = np.where(phi, 0, -1) + 0.5
 
    # compute the signed distance
    sdf = skfmm.distance(phi, dx =step) 
    return sdf

def GetDF(pts):
    distances = np.sqrt((pts[:,0][:,None] - X.ravel())**2 + (pts[:,1][:,None] - Y.ravel())**2)
    min_dist = distances.min(axis=0)
    return min_dist.reshape([dim,dim])


def ScaleAndSenter(contour):
    # Find max dist between pts
    max_dist = 0
    for a, b in combinations(np.array(contour),2):
        cur_dist = np.linalg.norm(a-b)
        if cur_dist > max_dist:
            max_dist = cur_dist
   
    # center and normalize contour
    center = np.mean(contour,axis=0)      
    contour = contour - center                 
    contour /= max_dist                     
    #ms = random.uniform(0.1,1.0) 
    ms = 0.5
    return contour, ms

def CreateData(dataType,i):
    P = GetRandomContour(nEdges)
 
    path = f'./data/{dataType}/{i}'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    P, ms = ScaleAndSenter(P)    

    # create mesh
    with pygmsh.geo.Geometry() as geom:
        #gmsh.option.setNumber('Mesh.RecombineAll', 1)
        geom.add_polygon(P, mesh_size=ms)
        mesh = geom.generate_mesh()
        mesh.write(f'{path}/mesh.vtk')
    mesh_pts = mesh.points.astype(np.float32)


    sdf = GetSignedDistance(P)    
    df = GetDF(mesh_pts[:,:-1])
    
    N = len(mesh_pts)

    sdf = torch.from_numpy(np.array(sdf)).view(1,dim,dim).float()
    df = torch.from_numpy(np.array(df)).view(1,dim,dim).float()
    N = torch.tensor(N).view(1,1,1).repeat(1,dim,dim).float() 

    data = {
        "P": P,
        "mesh_pts": mesh_pts,
        "sdf": sdf, 
        "df": df,
        "N_vert": N
    }    

    torch.save(data,f'{path}/data.pth')

# create training data for testing, training and validation
def CreateDataMain():
    for i in range(training_samples):
        if i < testing_samples:
            CreateData("test",i)
        if i < training_samples:
            CreateData("train",i)
        if i < validation_samples:
            CreateData("validation",i)
    print('Data sets created!')

# parameters for data
testing_samples = 1
training_samples = 1
validation_samples= 1
nEdges = 6
#ms = random.uniform(0.1,1.0) 
ms = 0.1

# create a grid
dim = 256
min_xy, max_xy = -0.8, 0.8
step = (max_xy - min_xy)/dim
xs = np.linspace(min_xy,max_xy,dim)
ys = np.linspace(min_xy,max_xy,dim)
X,Y = np.meshgrid(xs,ys)

if __name__ == "__main__":
    CreateDataMain()
