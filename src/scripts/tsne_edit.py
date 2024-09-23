import json
import os
from pathlib import Path

from os.path import join as pjoin

import numpy as np
import pytorch_lightning as pl
import torch
from rich.progress import track

from omegaconf import OmegaConf
from mld.data.utils import a2m_collate
from torch.utils.data import DataLoader
from mld.callback import ProgressLogger
from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger
import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.decomposition import TruncatedSVD
# from keras.datasets import mnist
from sklearn.datasets import load_iris
from numpy import reshape
import matplotlib
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mld.config import instantiate_from_config

params = {
'axes.labelsize': 36, # 20
'font.size': 18,
'legend.fontsize': 28, # 20
'xtick.labelsize': 20,
'ytick.labelsize': 20,
'text.usetex': False,
'figure.figsize': [2*4*1.5, 2*3*1.5]
}
rcParams.update(params)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
matplotlib.rc('pdf', fonttype=42)
matplotlib.font_manager._load_fontmanager(try_read_cache=False)

def data_parse(latents: np.ndarray, lengths, texts: list):

    # t_0 = latents[:,0,:].cpu() # [bs, 256]
    t_0 = latents.reshape(-1, latents.shape[-2] * latents.shape[-1]).cpu() # [bs, max_it*256]
    dim_red = TSNE(n_components=2, perplexity=70, random_state=140) # perplexity influnces quite a lot the result
    z = dim_red.fit_transform(t_0)
    
    # normalize
    z = 1.8*(z-np.min(z,axis=0))/(np.max(z,axis=0)-np.min(z,axis=0)) -0.9
    
    df = pd.DataFrame()
    df["y"] = np.array(texts)
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    df["length"] = np.array(lengths)
    return df

def data_parse3D(latents: np.ndarray, lengths, texts: list, length_aware=True):

    '''        
    lengths = np.array(lengths)
    
    points_for_X = latents[lengths >= 30, 0, :].cpu() #90
    points_for_Y = latents[lengths >= 96, 1, :].cpu() #60
    points_for_Z = latents[lengths >= 144, 2, :].cpu() #30

    # 1st method: do TSNE on concatenation of 1st, 2nd and 3rd latent, then split to recover X, Y and Z
    points_for_ALL = np.concatenate((points_for_X, points_for_Y, points_for_Z), axis=0)
    t_0 = points_for_ALL.reshape(-1, points_for_ALL.shape[-1]) # [bs, 1*256]
    dim_red = TSNE(n_components=1, perplexity=85, random_state=803)
    out = dim_red.fit_transform(t_0) # shape = [180, 1]

    # split the 3 points
    X = out[:len(points_for_X)]
    Y = out[len(points_for_X):len(points_for_X)+len(points_for_Y)]
    Z = out[len(points_for_X)+len(points_for_Y):]

    zeros = np.zeros_like(X)
    zeros[lengths >= 96] = Y
    Y = zeros

    zeros = np.zeros_like(X)
    zeros[lengths >= 144] = Z
    Z = zeros
    '''

    points_for_X = latents[:, 0, :].cpu() #90
    points_for_Y = latents[:, 1, :].cpu()
    points_for_Z = latents[:, 2, :].cpu()
    p = 47
    if not length_aware:
        p = 55
    rs = 803

    # 2nd method: do TSNE on 1st latents to extract X, 2nd latents to extract Y, 3rd latents to extract Z
    ########### X ###########
    t_0 = points_for_X.reshape(-1, points_for_X.shape[-1]) # [bs, 1*256]
    dim_red = TSNE(n_components=1, perplexity=p, random_state=rs) # perplexity influnces quite a lot the result 47 803 paper / 15 803 teaser
    X = dim_red.fit_transform(t_0)
    
    # normalize
    # X = 1.8*(X-np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0)) - 0.9

    ########### Y ###########
    t_0 = points_for_Y.reshape(-1, points_for_Y.shape[-1]) # [bs, 1*256]
    dim_red = TSNE(n_components=1, perplexity=p, random_state=rs) # perplexity influnces quite a lot the result
    Y = dim_red.fit_transform(t_0)
    
    # normalize
    # Y = 1.8*(Y-np.min(Y,axis=0))/(np.max(Y,axis=0)-np.min(Y,axis=0)) - 0.9

    ########### Z ###########
    t_0 = points_for_Z.reshape(-1, points_for_Z.shape[-1]) # [bs, 1*256]
    dim_red = TSNE(n_components=1, perplexity=p, random_state=rs) # 140 # perplexity influnces quite a lot the result
    Z = dim_red.fit_transform(t_0)
    
    # normalize
    # Z = 1.8*(Z-np.min(Z,axis=0))/(np.max(Z,axis=0)-np.min(Z,axis=0)) - 0.9

    df = pd.DataFrame()
    df["y"] = np.array(texts)
    df["comp-1"] = X
    df["comp-2"] = Y
    df["comp-3"] = Z
    df["length"] = lengths
    return df

def drawFig(output_dir: str, latents: np.ndarray, lengths, texts = None):
    ''' 
    Draw the figure of t-SNE
    Parameters:
        output_dir: output directory
        latents: (12, 50, 50, 256)
        steps: list of diffusion steps to draw
        classids: list of class ids
            # 0: "warm_up",
            # 1: "walk",
            # 2: "run",
            # 3: "jump",
            # 4: "drink",
            # 5: "lift_dumbbell",
            # 6: "sit",
            # 7: "eat",
            # 8: "turn steering wheel",
            # 9: "phone",
            # 10: "boxing",
            # 11: "throw",
    '''

    fig, axs = plt.subplots(1, 1)

    df = data_parse(latents, lengths, texts)

    # hue decides the class
    sns.scatterplot(ax=axs, x="comp-1", y="comp-2", hue='y',
                    legend = True,
                    palette=sns.color_palette("hls"),
                    data=df)
    axs.set_xlim((-1, 1))
    axs.set_ylim((-1, 1))
    # axs.set(yticklabels=[])
    # axs.set(ylabel=None)
    # axs.tick_params(left=False)

    for idx in range(df.shape[0]):
        axs.text(df.iloc[idx]["comp-1"]+0.01, df.iloc[idx]["comp-2"], 
                df.iloc[idx]["length"], horizontalalignment='left', 
                size='medium', color='black')

    plt.legend(title='Text')
    #plt.legend([],[], frameon=False)
    plt.savefig(pjoin(output_dir, 'LATENTSPACE.png'), bbox_inches='tight')
    plt.close()

def drawFig3D(output_dir: str, latents: np.ndarray, lengths, texts = None, length_aware=True):
    ''' 
    Draw the figure of t-SNE
    Parameters:
        output_dir: output directory
        latents: output of backward diffusion of the model
        lengths: array of lengths of the motions
        texts: list of texts of the motions
        length_aware: if True, draw latent space of LA model
    '''

    df = data_parse3D(latents, lengths, texts, length_aware=length_aware)
    # il dataframe ha le seguenti colonne: 
    # y: testo
    # comp-1: x
    # comp-2: y
    # comp-3: z
    # length: lunghezza del motion

    class_colors = {
        df.iloc[0]["y"]+str(df.iloc[0]["length"]): 'salmon',         df.iloc[0]["y"]+str(df.iloc[1]["length"]): 'tab:red',            df.iloc[0]["y"]+str(df.iloc[2]["length"]): 'maroon',
        df.iloc[3]["y"]+str(df.iloc[0]["length"]): 'deepskyblue', df.iloc[3]["y"]+str(df.iloc[1]["length"]): 'tab:blue',           df.iloc[3]["y"]+str(df.iloc[2]["length"]): 'midnightblue',
        df.iloc[6]["y"]+str(df.iloc[0]["length"]): 'lime',           df.iloc[6]["y"]+str(df.iloc[1]["length"]): 'mediumseagreen', df.iloc[6]["y"]+str(df.iloc[2]["length"]): "darkgreen"
    }
    
    # filter one red outlier for paper figure
    mask = (df['comp-2'] == df[(df["length"] == 30) & (df["y"] == "a man jumps.")]["comp-2"].max()) & (df["length"] == 30)
    df = df[~mask]

    # figure instance
    fig = plt.figure(figsize=(3*2*4*1.5, 2*3*1.5))
    
    ax1 = fig.add_subplot(131)
    # plot elements in datafram that have only x component
    for_x = df[df["length"] == 30]
    colors_x = [class_colors[i] for i in for_x["y"]+for_x["length"].astype(str)] 
    ax1.scatter(for_x["comp-1"], for_x["comp-2"], c=colors_x, s=50, label=for_x["y"])
    ax1.set(xlabel="X", ylabel="Y")
    ax1.legend().remove()

    ax2 = fig.add_subplot(132)
    # plot elements in datafram that have x,y components
    for_xy = df[df["length"] <= 96]
    colors_x = [class_colors[i] for i in for_xy["y"]+for_xy["length"].astype(str)] 
    ax2.scatter(for_xy["comp-1"], for_xy["comp-2"], c=colors_x, s=50, label=for_xy["y"])
    ax2.set(xlabel="X", ylabel="Y")
    ax2.legend().remove()

    axs3D = fig.add_subplot(133, projection='3d')
    axs3D.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    axs3D.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    axs3D.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # plot all
    axs3D.set_box_aspect((1,1,1), zoom=1.55)

    # Plot the data points with colors and length labels
    for i in range(len(df)):
        x, y, z = df.iloc[i][['comp-1', 'comp-2', 'comp-3']]
        class_label = df.iloc[i]['y']
        l = str(df.iloc[i]["length"])
        axs3D.scatter(x, y, z, c=class_colors[class_label+l], s=50, label=class_label if l == "96" else None)
    # axs.set_xlim((-1, 1))
    # axs.set_ylim((-1, 1))
    # axs.set_zlim((-1, 1))
    axs3D.yaxis.set_ticklabels([])
    axs3D.yaxis._axinfo['tick']['inward_factor'] = 0.0
    axs3D.yaxis._axinfo['tick']['outward_factor'] = 0.0
    axs3D.set(xlabel="X", ylabel="Y")
    axs3D.set_zlabel("Z", rotation=90)
    axs3D.xaxis.labelpad=0
    axs3D.zaxis.labelpad=40
    axs3D.xaxis.set_tick_params(pad=-4)
    axs3D.zaxis.set_tick_params(pad=20)
    axs3D.view_init(elev=0, azim=-90)

    # plt.legend(title='Text')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    axs3D.legend().remove()

    plt.savefig(pjoin(output_dir, 'multiview.png'), bbox_inches='tight')
    plt.close()

    # multiview saved, now draw 3D latent space

    fig = plt.figure()

    axs3D = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(axs3D)
    axs3D.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    axs3D.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    axs3D.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    for i in range(len(df)):
        x, y, z = df.iloc[i][['comp-1', 'comp-2', 'comp-3']]
        class_label = df.iloc[i]['y']
        l = str(df.iloc[i]["length"])
        axs3D.scatter(x, y, z, c=class_colors[class_label+l], s=50, label=class_label if l == "96" else None)
    # axs.set_xlim((-1, 1))
    # axs.set_ylim((-1, 1))
    # axs.set_zlim((-1, 1))
    axs3D.set(xlabel="X", ylabel="Y", zlabel="Z")
    axs3D.xaxis.labelpad=20
    axs3D.yaxis.labelpad=20
    axs3D.zaxis.set_tick_params(pad=20)
    axs3D.zaxis.labelpad=35

    if length_aware:
        # draw lines
        edge = 0.14

        line_blue = df[(df["length"] == 30) & (df["y"] == "a man runs.")]
        line_green = df[(df["length"] == 30) & (df["y"] == "a man sits.")]
        line_red = df[(df["length"] == 30) & (df["y"] == "a man jumps.")]

        color = class_colors[line_blue.iloc[0]["y"]+str(line_blue.iloc[0]["length"])]
        axs3D.plot([line_blue["comp-1"].min()-edge, (line_blue["comp-1"].max() + line_green["comp-1"].min()) / 2],
                    [line_blue["comp-2"].min(), line_blue["comp-2"].min()],
                    [line_blue["comp-3"].min(), line_blue["comp-3"].min()], color=color, linewidth=2)
        
        color = class_colors[line_green.iloc[0]["y"]+str(line_green.iloc[0]["length"])]
        axs3D.plot([(line_blue["comp-1"].max() + line_green["comp-1"].min()) / 2, (line_green["comp-1"].max() + line_red["comp-1"].min()) / 2],
                    [line_green["comp-2"].min(), line_green["comp-2"].min()],
                    [line_green["comp-3"].min(), line_green["comp-3"].min()], color=color, linewidth=2)    
        
        color = class_colors[line_red.iloc[0]["y"]+str(line_red.iloc[0]["length"])]
        axs3D.plot([(line_green["comp-1"].max() + line_red["comp-1"].min()) / 2, line_red["comp-1"].max()+edge],
                    [line_red["comp-2"].min(), line_red["comp-2"].min()],
                    [line_red["comp-3"].min(), line_red["comp-3"].min()], color=color, linewidth=2)

        #######################################################
        # draw hyperplanes
        
        surface_red = df[(df["length"] == 96) & (df["y"] == "a man jumps.")]
        xx, yy = np.meshgrid([surface_red["comp-1"].max()+edge, surface_red["comp-1"].min()-edge],
                            [surface_red["comp-2"].max()+edge, surface_red["comp-2"].min()-edge])
        zz = np.full(xx.shape, surface_red["comp-3"].min())  # Z coordinates are all constant
        color = class_colors[surface_red.iloc[0]["y"]+str(surface_red.iloc[0]["length"])]
        axs3D.plot_surface(xx, yy, zz, color=color, alpha=0.2) # Plot the rectangular hyperplane using plot_surface
        
        surface_blue = df[(df["length"] == 96) & (df["y"] == "a man runs.")]
        xx, yy = np.meshgrid([surface_blue["comp-1"].max()+edge, surface_blue["comp-1"].min()-edge],
                            [surface_blue["comp-2"].max()+edge, surface_blue["comp-2"].min()-edge])
        zz = np.full(xx.shape, surface_blue["comp-3"].min())  # Z coordinates are all constant
        color = class_colors[surface_blue.iloc[0]["y"]+str(surface_blue.iloc[0]["length"])]
        axs3D.plot_surface(xx, yy, zz, color=color, alpha=0.2) # Plot the rectangular hyperplane using plot_surface

        surface_green = df[(df["length"] == 96) & (df["y"] == "a man sits.")]
        xx, yy = np.meshgrid([surface_green["comp-1"].max()+edge, surface_green["comp-1"].min()-edge],
                            [surface_green["comp-2"].max()+edge, surface_green["comp-2"].min()-edge])
        zz = np.full(xx.shape, surface_green["comp-3"].min())  # Z coordinates are all constant
        color = class_colors[surface_green.iloc[0]["y"]+str(surface_green.iloc[0]["length"])]
        axs3D.plot_surface(xx, yy, zz, color=color, alpha=0.2) # Plot the rectangular hyperplane using plot_surface

        #######################################################
        # draw parallelepipeds
        edge = 0.2 / 3

        parallelepiped_red = df[(df["length"] == 144) & (df["y"] == "a man jumps.")]
        points = np.array([[parallelepiped_red["comp-1"].min()-edge, parallelepiped_red["comp-2"].min()-edge, parallelepiped_red["comp-3"].min()-edge],
                        [parallelepiped_red["comp-1"].max()+edge, parallelepiped_red["comp-2"].min()-edge, parallelepiped_red["comp-3"].min()-edge],
                        [parallelepiped_red["comp-1"].max()+edge, parallelepiped_red["comp-2"].max()+edge, parallelepiped_red["comp-3"].min()-edge],
                        [parallelepiped_red["comp-1"].min()-edge, parallelepiped_red["comp-2"].max()+edge, parallelepiped_red["comp-3"].min()-edge],
                        [parallelepiped_red["comp-1"].min()-edge, parallelepiped_red["comp-2"].min()-edge, parallelepiped_red["comp-3"].max()+edge],
                        [parallelepiped_red["comp-1"].max()+edge, parallelepiped_red["comp-2"].min()-edge, parallelepiped_red["comp-3"].max()+edge],
                        [parallelepiped_red["comp-1"].max()+edge, parallelepiped_red["comp-2"].max()+edge, parallelepiped_red["comp-3"].max()+edge],
                        [parallelepiped_red["comp-1"].min()-edge, parallelepiped_red["comp-2"].max()+edge, parallelepiped_red["comp-3"].max()+edge]])

        # list of sides' polygons of figure
        verts = [[points[0],points[1],points[2],points[3]],
                [points[4],points[5],points[6],points[7]], 
                [points[0],points[1],points[5],points[4]], 
                [points[2],points[3],points[7],points[6]], 
                [points[1],points[2],points[6],points[5]],
                [points[4],points[7],points[3],points[0]]]

        # plot sides
        color = class_colors[parallelepiped_red.iloc[0]["y"]+str(parallelepiped_red.iloc[0]["length"])]
        axs3D.add_collection3d(Poly3DCollection(verts, facecolors=color, alpha=0.2))

        parallelepiped_blue = df[(df["length"] == 144) & (df["y"] == "a man runs.")]
        points = np.array([[parallelepiped_blue["comp-1"].min()-edge, parallelepiped_blue["comp-2"].min()-edge, parallelepiped_blue["comp-3"].min()-edge],
                        [parallelepiped_blue["comp-1"].max()+edge, parallelepiped_blue["comp-2"].min()-edge, parallelepiped_blue["comp-3"].min()-edge],
                        [parallelepiped_blue["comp-1"].max()+edge, parallelepiped_blue["comp-2"].max()+edge, parallelepiped_blue["comp-3"].min()-edge],
                        [parallelepiped_blue["comp-1"].min()-edge, parallelepiped_blue["comp-2"].max()+edge, parallelepiped_blue["comp-3"].min()-edge],
                        [parallelepiped_blue["comp-1"].min()-edge, parallelepiped_blue["comp-2"].min()-edge, parallelepiped_blue["comp-3"].max()+edge],
                        [parallelepiped_blue["comp-1"].max()+edge, parallelepiped_blue["comp-2"].min()-edge, parallelepiped_blue["comp-3"].max()+edge],
                        [parallelepiped_blue["comp-1"].max()+edge, parallelepiped_blue["comp-2"].max()+edge, parallelepiped_blue["comp-3"].max()+edge],
                        [parallelepiped_blue["comp-1"].min()-edge, parallelepiped_blue["comp-2"].max()+edge, parallelepiped_blue["comp-3"].max()+edge]])

        # list of sides' polygons of figure
        verts = [[points[0],points[1],points[2],points[3]],
                [points[4],points[5],points[6],points[7]], 
                [points[0],points[1],points[5],points[4]], 
                [points[2],points[3],points[7],points[6]], 
                [points[1],points[2],points[6],points[5]],
                [points[4],points[7],points[3],points[0]]]

        # plot sides
        color = class_colors[parallelepiped_blue.iloc[0]["y"]+str(parallelepiped_blue.iloc[0]["length"])]
        axs3D.add_collection3d(Poly3DCollection(verts, facecolors=color, alpha=0.2))
        
        parallelepiped_green = df[(df["length"] == 144) & (df["y"] == "a man sits.")]
        points = np.array([[parallelepiped_green["comp-1"].min()-edge, parallelepiped_green["comp-2"].min()-edge, parallelepiped_green["comp-3"].min()-edge],
                        [parallelepiped_green["comp-1"].max()+edge, parallelepiped_green["comp-2"].min()-edge, parallelepiped_green["comp-3"].min()-edge],
                        [parallelepiped_green["comp-1"].max()+edge, parallelepiped_green["comp-2"].max()+edge, parallelepiped_green["comp-3"].min()-edge],
                        [parallelepiped_green["comp-1"].min()-edge, parallelepiped_green["comp-2"].max()+edge, parallelepiped_green["comp-3"].min()-edge],
                        [parallelepiped_green["comp-1"].min()-edge, parallelepiped_green["comp-2"].min()-edge, parallelepiped_green["comp-3"].max()+edge],
                        [parallelepiped_green["comp-1"].max()+edge, parallelepiped_green["comp-2"].min()-edge, parallelepiped_green["comp-3"].max()+edge],
                        [parallelepiped_green["comp-1"].max()+edge, parallelepiped_green["comp-2"].max()+edge, parallelepiped_green["comp-3"].max()+edge],
                        [parallelepiped_green["comp-1"].min()-edge, parallelepiped_green["comp-2"].max()+edge, parallelepiped_green["comp-3"].max()+edge]])

        # list of sides' polygons of figure
        verts = [[points[0],points[1],points[2],points[3]],
                [points[4],points[5],points[6],points[7]], 
                [points[0],points[1],points[5],points[4]], 
                [points[2],points[3],points[7],points[6]], 
                [points[1],points[2],points[6],points[5]],
                [points[4],points[7],points[3],points[0]]]

        # plot sides
        color = class_colors[parallelepiped_green.iloc[0]["y"]+str(parallelepiped_green.iloc[0]["length"])]
        axs3D.add_collection3d(Poly3DCollection(verts, facecolors=color, alpha=0.2))

    #######################################################
    # plt.legend(title='Text')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # axs3D.set_box_aspect(None, zoom=1.05)
    plt.savefig(pjoin(output_dir, 'hyperplanes.png'))

    def rotate(angle):
        axs3D.view_init(azim=angle)

    # render rotating gif
    # rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 361))
    # rot_animation.save(pjoin(output_dir, 'rotation.gif'), fps=60)

    plt.close()
    

def drawTeaser(output_dir: str, latents: np.ndarray, lengths, texts = None, length_aware=True):
    ''' 
    Draw the figure of t-SNE
    Parameters:
        output_dir: output directory
        latents: (12, 50, 50, 256)
        steps: list of diffusion steps to draw
        classids: list of class ids
            # 0: "warm_up",
            # 1: "walk",
            # 2: "run",
            # 3: "jump",
            # 4: "drink",
            # 5: "lift_dumbbell",
            # 6: "sit",
            # 7: "eat",
            # 8: "turn steering wheel",
            # 9: "phone",
            # 10: "boxing",
            # 11: "throw",
    '''

    df = data_parse3D(latents, lengths, texts, length_aware=length_aware)
    class_colors = {
        df.iloc[0]["y"]+str(df.iloc[0]["length"]): 'limegreen',
        df.iloc[1]["y"]+str(df.iloc[1]["length"]): 'mediumseagreen',
        df.iloc[2]["y"]+str(df.iloc[2]["length"]): "darkgreen"
    }

    # remove points of length 30 that have y,z value different from majority
    mask = (df['comp-2'] != df[(df["length"] == 30)]["comp-2"].mode().item()) & (df["length"] == 30)
    df = df[~mask]
    mask = (df['comp-3'] != df[(df["length"] == 30)]["comp-3"].mode().item()) & (df["length"] == 30)
    df = df[~mask]
    # remove points of length 96 that have z value different from majority
    mask = (df['comp-3'] != df[(df["length"] == 96)]["comp-3"].mode().item()) & (df["length"] == 96)
    df = df[~mask]

    # new figure instance
    fig = plt.figure()

    axs3D = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(axs3D)
    axs3D.set_axis_off()

    for i in range(len(df)):
        x, y, z = df.iloc[i][['comp-1', 'comp-2', 'comp-3']]
        class_label = df.iloc[i]['y']
        l = str(df.iloc[i]["length"])
        axs3D.scatter(x, y, z, c=class_colors[class_label+l], s=50)

    #######################################################

    if length_aware:
        # draw line
        edge = 0.2

        line_green = df[(df["length"] == 30) & (df["y"] == "a man sits.")]
        
        color = class_colors[line_green.iloc[0]["y"]+str(line_green.iloc[0]["length"])]
        axs3D.plot([df["comp-1"].min()-edge, df["comp-1"].max()+edge],
                    [line_green["comp-2"].max(), line_green["comp-2"].max()],
                    [line_green["comp-3"].min(), line_green["comp-3"].min()], color=color, linewidth=2)    

        #######################################################

        # draw hyperplane
        
        surface_green = df[(df["length"] == 96) & (df["y"] == "a man sits.")]
        xx, yy = np.meshgrid([df["comp-1"].max()+edge, df["comp-1"].min()-edge],
                            [df["comp-2"].max()+edge, df["comp-2"].min()-edge])
        zz = np.full(xx.shape, surface_green["comp-3"].min())  # Z coordinates are all constant
        color = class_colors[surface_green.iloc[0]["y"]+str(surface_green.iloc[0]["length"])]
        axs3D.plot_surface(xx, yy, zz, color=color, edgecolor=color, alpha=0.2) # Plot the rectangular hyperplane using plot_surface
        #######################################################
        
        # draw parallelepipeds
     
        parallelepiped_green = df[(df["length"] == 144) & (df["y"] == "a man sits.")]
        points = np.array([[df["comp-1"].min()-edge, df["comp-2"].min()-edge, df["comp-3"].min()-edge],
                        [df["comp-1"].max()+edge, df["comp-2"].min()-edge, df["comp-3"].min()-edge],
                        [df["comp-1"].max()+edge, df["comp-2"].max()+edge, df["comp-3"].min()-edge],
                        [df["comp-1"].min()-edge, df["comp-2"].max()+edge, df["comp-3"].min()-edge],
                        [df["comp-1"].min()-edge, df["comp-2"].min()-edge, df["comp-3"].max()+edge],
                        [df["comp-1"].max()+edge, df["comp-2"].min()-edge, df["comp-3"].max()+edge],
                        [df["comp-1"].max()+edge, df["comp-2"].max()+edge, df["comp-3"].max()+edge],
                        [df["comp-1"].min()-edge, df["comp-2"].max()+edge, df["comp-3"].max()+edge]])

        # list of sides' polygons of figure
        verts = [[points[0],points[1],points[2],points[3]],
                [points[4],points[5],points[6],points[7]], 
                [points[0],points[1],points[5],points[4]], 
                [points[2],points[3],points[7],points[6]], 
                [points[1],points[2],points[6],points[5]],
                [points[4],points[7],points[3],points[0]]]

        # plot sides
        color = class_colors[parallelepiped_green.iloc[0]["y"]+str(parallelepiped_green.iloc[0]["length"])]
        axs3D.add_collection3d(Poly3DCollection(verts, facecolors=color, edgecolor=color, alpha=0.2))
    #######################################################
    
    # for idx in range(len(df)):
    #     axs.text(df.iloc[idx]["comp-1"]+0.01, df.iloc[idx]["comp-2"], df.iloc[idx]["comp-3"],
    #             df.iloc[idx]["length"], horizontalalignment='left', 
    #             size='small', c='black')

    axs3D.view_init(elev=20, azim=-80)
    plt.savefig(pjoin(output_dir, 'teaser.png'), bbox_inches='tight')

    def rotate(angle):
        axs3D.view_init(azim=angle)

    # rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 361))
    # rot_animation.save(pjoin(output_dir, 'rotation.gif'), fps=60)

    plt.close()


def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.FOLDER = cfg.TEST.FOLDER

    # create logger
    logger = create_logger(cfg, phase="test")
    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
                     "tsne_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(OmegaConf.to_yaml(cfg))

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        #     str(x) for x in cfg.DEVICE)
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # create dataset
    dataset = get_datasets(cfg, logger=logger, phase="test")[0]
    logger.info("datasets module {} initialized".format("".join(
        cfg.TRAIN.DATASETS)))
    subset = 'train'.upper() 
    split = eval(f"cfg.{subset}.SPLIT")
    split_file = pjoin(
                    eval(f"cfg.DATASET.{dataset.name.upper()}.SPLIT_ROOT"),
                    eval(f"cfg.{subset}.SPLIT") + ".txt",
                )
    dataloader = DataLoader(dataset.Dataset(split_file=split_file,split=split,**dataset.hparams),batch_size=8,collate_fn=a2m_collate)

    # create model
    model = get_model(cfg, dataset)
    logger.info("model {} loaded".format(cfg.model.model_type))

    # loading state dict
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))

    state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                            map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)
    model = model.eval()

    text_encoder = instantiate_from_config(cfg.model.text_encoder)
    text_encoder = text_encoder.eval()

    # Device
    if cfg.ACCELERATOR == "gpu":
        device = torch.device("cuda:6")
        model = model.to(device)
        text_encoder = text_encoder.to(device)
    
    
    # Generate latent codes
    with torch.no_grad():
        print('Doing inference...')
        # MODIFY THESE TWO IF YOU WANT TO TEST OTHER MOTIONS
        orig_texts = ['a man jumps.', 'a man jumps.', 'a man jumps.',
                    'a man runs.', 'a man runs.', 'a man runs.',
                    'a man sits.', 'a man sits.', 'a man sits.']*10
        lengths = [30, 96, 144, 30, 96, 144, 30, 96, 144]*10

        # for cfg
        uncond_tokens = [""] * len(orig_texts)
        uncond_tokens.extend(orig_texts)
        texts = uncond_tokens
        cond_emb = text_encoder(texts) # [bs, emb_dim]
        
        z = model._diffusion_reverse(cond_emb, lengths)
        z = z.permute(1, 0, 2) # [latent_dim[0], bs, latent_dim[1]] -> [bs, latent_dim[0], latent_dim[1]]
        #print(z.shape)
        
    # Draw figure
    print('Drawing figure...')
    #drawFig(output_dir, z, lengths, texts = orig_texts)
    drawFig3D(output_dir, z, lengths, texts = orig_texts, length_aware=True)
    
    # Draw teaser
    with torch.no_grad():
        print('Doing inference for teaser...')
        # MODIFY THESE TWO IF YOU WANT TO TEST OTHER MOTIONS
        orig_texts = ['a man sits.', 'a man sits.', 'a man sits.']*15
        lengths = [30, 96, 144]*15

        # for cfg
        uncond_tokens = [""] * len(orig_texts)
        uncond_tokens.extend(orig_texts)
        texts = uncond_tokens
        cond_emb = text_encoder(texts) # [bs, emb_dim]
        
        z = model._diffusion_reverse(cond_emb, lengths)
        z = z.permute(1, 0, 2) # [latent_dim[0], bs, latent_dim[1]] -> [bs, latent_dim[0], latent_dim[1]]


    # print('Drawing teaser...')
    # drawTeaser(output_dir, z, lengths, texts = orig_texts)

    logger.info("TSNE figures saved to {}".format(output_dir))

if __name__ == "__main__":
    main()