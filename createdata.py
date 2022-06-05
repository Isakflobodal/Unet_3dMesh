import numpy as np
import matplotlib.pyplot as plt
import random
import pygmsh
import os
import torch
from itertools import combinations
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pyvista as pv
from PIL import Image, ImageDraw
import skfmm
import meshpy
import trimesh
from pysdf import SDF

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
    return contour

def GetRandomContour(N):
    l = 1.0
    b = random.uniform(0.5,1.0)
    h = random.uniform(0.5,1.0)

    contour = np.array([[-l/2,-b/2,-h/2],[l/2,-b/2,-h/2],[l/2,b/2,-h/2],[-l/2,b/2,-h/2],[-l/2,-b/2,h/2],[l/2,-b/2,h/2],[l/2,b/2,h/2],[-l/2,b/2,h/2]])
    contour = ScaleAndSenter(contour=contour)
    l = abs(contour[0][0])*2
    b = abs(contour[0][1])*2
    h = abs(contour[0][2])*2

    target_len = np.cbrt(b*h*l/N)
    nl = max(round(l/target_len)+1,2)
    nb = max(round(b/target_len)+1,2)
    nh = max(round(h/target_len)+1,2)

    ls = np.linspace(-l/2,l/2,nl)
    bs = np.linspace(-b/2,b/2,nb)
    hs = np.linspace(-h/2,h/2,nh)

    X,Y = np.meshgrid(ls,bs)
    pts = np.array([list(pair) for pair in zip(X.flatten(),Y.flatten(), np.full(nl*nb,0))])

    levels = []
    for l in hs:
        level = pts.copy()
        level[:,-1] = l
        levels.append(level)
    
    mesh = pv.StructuredGrid()
    pts = np.vstack(levels)
    mesh.points = pts
    mesh.dimensions = [nl,nb,nh]
    surf_mesh = trimesh.primitives.Box((l,b,h))

    #mesh.plot(show_edges=True, show_grid=True, cpos = 'xy')

    return pts, contour, mesh, surf_mesh

# Returns a distance field
def GetDF(pts):
    x = (pts[:,0][:,None] - BB[:,0])
    y = (pts[:,1][:,None] - BB[:,1])
    z = (pts[:,2][:,None] - BB[:,2])
    #print(x.shape)
    vec = np.vstack([x[None],y[None],z[None]]).swapaxes(0,2)  # [P,36,3]
    vec_length = np.linalg.norm(vec, axis=-1) # [P,36]
    a = np.arange(0, vec_length.shape[0])
    
    min_vec_length_idx = np.argmin(vec_length, axis=1) # [P]
    min_length = vec_length[a, min_vec_length_idx]
    min_vec = vec[a,min_vec_length_idx]
    #print(min_vec.shape, min_length.shape)

    # print(vec.shape)
    # distances = np.sqrt(x**2+y**2+z**2)
    # min_dist = distances.min(axis=0)
    return min_length.reshape([dim,dim,dim]), min_vec.reshape([dim,dim,dim,3])


def Sdf3D(surf_mesh):
    f = SDF(surf_mesh.vertices, surf_mesh.faces)
    sdf = f(BB.tolist())
    #sdf = mesh_to_sdf(surf_mesh,BB)
    return sdf

# Create contours, mesh and distance field
def CreateData(dataType,i, N):
    path = f'./data/{dataType}/{i}'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)     

    mesh_pts, P, mesh, surf_mesh = GetRandomContour(N=N)       
    mesh_pts = np.array(mesh_pts) 
    pv.save_meshio(f'{path}/mesh.vtk', mesh=mesh)


    sdf = Sdf3D(surf_mesh)
    sdf = torch.from_numpy(np.array(sdf)).view(dim,dim,dim).float()
    df, df_vec = GetDF(mesh_pts)
    df = torch.from_numpy(np.array(df)).float()
    df_vec = torch.from_numpy(np.array(df_vec)).float()
    
    data = {
        "Pc": P,
        "mesh_pts": mesh_pts,
        "df": df,
        "df_vec": df_vec,
        'sdf':sdf
        }

    torch.save(data,f'{path}/data.pth')
    return sdf, df, mesh_pts, P

# create training data for testing, training and validation
def CreateDataMain(N):
    for j in range(training_samples):
        i = j + 5533
        if i < testing_samples:
            CreateData("test",i,N)
        if i < training_samples:
            sdf, df, mesh_pts, P = CreateData("train",i, N)
            df_list.append(df)
            mesh_pts_list.append(mesh_pts)
            sdf_list.append(sdf)
            P_list.append(P)
        if i < validation_samples:
            CreateData("validation",i, N)
    print('Data sets created!')


# Plots the distance field
def PlotDistanceField(df, sdf, mesh_pts, P):

    x, y, z = BB[:,0], BB[:,1], BB[:,2]
    max, min = x.max().item(), x.min().item()
    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(1,2,1, projection = '3d')
    ax.scatter(x,y,z, s=30, c = df, cmap='RdBu', alpha=0.03)
    ax.set_title('Distance function')
    ax.scatter(mesh_pts[:,0],mesh_pts[:,1],mesh_pts[:,2], c='green' )
    ax.set_xlim(min,max)
    ax.set_ylim(min,max)
    ax.set_zlim(min,max)

    ax = fig.add_subplot(1,2,2, projection = '3d')
    ax.scatter(x,y,z, s=30, c = sdf, cmap='hsv', alpha=0.03)
    ax.scatter(P[:,0],P[:,1],P[:,2], c='red' )
    ax.set_title('Signed distance field')
    ax.set_xlim(min,max)
    ax.set_ylim(min,max)
    ax.set_zlim(min,max)
    fig.tight_layout()
    plt.show()


# Grid size
dim = 64
min_xy, max_xy = -0.5, 0.5
step = (max_xy - min_xy)/dim
xs = np.linspace(min_xy,max_xy,dim)
ys = np.linspace(min_xy,max_xy,dim)
zs = np.linspace(min_xy, max_xy,dim)
X,Y = np.meshgrid(xs,ys)

pts = np.array([list(pair) for pair in zip(X.flatten(),Y.flatten(), np.full(dim*dim,0))])

BB = []
for l in zs:
    level = pts.copy()
    level[:,-1] = l
    BB.append(level)
BB = np.array(BB)
BB = BB.reshape(-1,3)

# Hyperparameters for data
testing_samples = 500
training_samples = 10000
validation_samples= 1000
N = 10

df_list = []
sdf_list = []
mesh_pts_list = []
P_list = []


if __name__ == "__main__":
    CreateDataMain(N=N)
    PlotDistanceField(df_list[0], sdf_list[0], mesh_pts_list[0], P_list[0])