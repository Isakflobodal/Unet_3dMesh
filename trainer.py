import matplotlib.pyplot as plt
from NN import NeuralNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.autograd as ag

from createdata import X,Y
import tqdm
import wandb
from torchvision.utils import make_grid
from createdata import BB
from matplotlib import cm
import numpy as np
PTS = torch.from_numpy(BB).float()
pts = PTS
colormap = cm.get_cmap('RdBu',12)

@torch.no_grad()
def no_grad_loop(data_loader, model, png_cnt, epoch=2, device="cuda", batch_size = 64):
    valid_loss = 0
    cnt = 0
    for i, (Pc, df, sdf) in enumerate(data_loader):

        # transfer data to device
        Pc = Pc.to(device)
        df = df.to(device)
        sdf = sdf.to(device)

        with autocast():
            df_pred = model(sdf)
            loss = F.l1_loss(df_pred, df)
        valid_loss += loss
        cnt += 1

        if i == len(data_loader) -1:
            case = 2
            x, y, z = pts[:,0], pts[:,1], pts[:,2]
            max, min = x.max().item(), x.min().item()
            
            #df = torch.linalg.norm(df, dim=-1)
            #df_pred = torch.linalg.norm(df_pred, dim=-1)
            df = df[case].cpu().squeeze().detach().numpy()
            df_pred = df_pred[case].cpu().squeeze().detach().numpy()

            fig = plt.figure(figsize=plt.figaspect(0.5))

            ax = fig.add_subplot(1,2,1, projection = '3d')
            ax.scatter(x,y,z, s=30, c = df, cmap = 'RdBu', alpha=0.005)
            ax.set_title('Target')
            ax.set_xlim(min,max)
            ax.set_ylim(min,max)
            ax.set_zlim(min,max)

            ax = fig.add_subplot(1,2,2, projection = '3d')
            ax.scatter(x,y,z, s=30, c = df_pred, cmap = 'RdBu', alpha=0.005)
            ax.set_title('Prediction')
            ax.set_xlim(min,max)
            ax.set_ylim(min,max)
            ax.set_zlim(min,max)

            if png_cnt != 100000000:
                plt.savefig(f'./GIFS_training/pngs/{png_cnt}.png')

            plt.savefig('train.png')
            


            # pointcloud at wandb
            # max_value = df.max().item()
            # df_scaled = df/max_value 
            # color_list = torch.tensor(colormap(df_scaled)*255).view(-1,4)
            # pts_rgba = torch.cat((pts,color_list), dim=1)
            # pt_cloud = np.array(pts_rgba[:,:-1].tolist())    
            
            # target_wandb = wandb.Object3D({
            #     "type": "lidar/beta",
            #     "points": np.array(pt_cloud),
            #     "boxes": np.array(
            #         [
            #             {
            #                 "corners": Pc[case].cpu().squeeze().detach().numpy().tolist(),
            #                 "label": "Box",
            #                 "color": [250,2,233],
            #             }
            #         ]
            #     ),
            # })

            

            # max_value = df_pred.max().item()
            # df_scaled = df_pred/max_value 
            # color_list = torch.tensor(colormap(df_scaled)*255).view(-1,4)
            # pts_rgba = torch.cat((pts,color_list), dim=1)
            # pt_cloud = np.array(pts_rgba[:,:-1].tolist())    
            
            # pred_wandb = wandb.Object3D({
            #     "type": "lidar/beta",
            #     "points": np.array(pt_cloud),
            #     "boxes": np.array(
            #         [
            #             {
            #                 "corners": Pc[case].cpu().squeeze().detach().numpy().tolist(),
            #                 "label": "Box",
            #                 "color": [250,2,233],
            #             }
            #         ]
            #     ),
            # })


            #wandb.log({"Images": wandb.Image("train.png"),"Point cloud target": target_wandb, "Point cloud prediction": pred_wandb }, commit=False)
            wandb.log({"Images": wandb.Image("train.png")}, commit=False)
            plt.close(fig)
            plt.close('all')

    return valid_loss/cnt

def train(model: NeuralNet, num_epochs, batch_size, train_loader, test_loader, validation_loader, learning_rate=1e-3, device="cuda"):

    valid_loss = no_grad_loop(validation_loader, model, png_cnt=0, epoch=0, device="cuda", batch_size=batch_size)
    curr_lr =  learning_rate

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=6)

    # early stopping
    last_loss = 100000000
    best_loss = 100000000
    patience = 10
    trigger_times = 0


    # training loop
    iter = 0
    png_cnt = 1
    training_losses = {
        "train": {},
        "valid": {}
    }
    scaler = GradScaler()
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        loader = tqdm.tqdm(train_loader)
        for _, df, sdf in loader:  

            # transfer data to device
            df = df.to(device)
            sdf = sdf.to(device)
            
            # Forward pass
            with autocast():
                df_pred = model(sdf) 
                loss = F.l1_loss(df_pred, df)
            loader.set_postfix(df_loss = loss.item())
            
            # Backward and optimize
            optimizer.zero_grad()               # clear gradients
            scaler.scale(loss).backward()       # calculate gradients

            # grad less than 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)

            scaler.step(optimizer)
            scaler.update()    
            iter += 1  

            training_losses["train"][iter] = loss.item()
            if (iter+1) %  20 == 0:

                # validation loop
                model = model.eval()
                valid_loss = no_grad_loop(validation_loader, model, png_cnt, epoch, device="cuda", batch_size=batch_size)

                #early stopping
                if valid_loss > last_loss:
                    trigger_times +=1
                    last_loss = valid_loss
                    if trigger_times >= patience:
                        return model
                else:
                    trigger_times = 0
                #last_loss = valid_loss

                # save check point
                if valid_loss < best_loss:
                    config = {"state_dict": model.state_dict()}
                    torch.save(config, "model_check_point.pth")
                    best_loss = valid_loss



                png_cnt += 1
                scheduler.step(valid_loss)
                curr_lr =  optimizer.param_groups[0]["lr"]
                wandb.log({"valid loss": valid_loss.item(), "lr": curr_lr}, commit=False)
                model = model.train()
            wandb.log({"train loss": loss.item()})

    # test loop
    test_loss = no_grad_loop(test_loader, model, png_cnt=100000000, device="cuda", batch_size=batch_size)
    print(f'Testloss: {test_loss:.5f}')
    return model