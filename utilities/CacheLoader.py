import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def Params_to_Filename(params, keylist = []):
    file_name = ""
    if len(keylist) <= 0:
        for key, item in params.items():
            file_name += '_' if len(file_name) > 0 else ''
            file_name += key + '=' + str(item)
    else:
        for key in keylist:
            file_name += '_' if len(file_name) > 0 else ''
            file_name += key + '=' + str(params[key])
    return file_name

def Check_and_Load(file_names):
    ifexists = []
    loaded = []
    for file_name in file_names:
        ok = os.path.exists(file_name + '.npy')
        ifexists.append(ok)
        if ok:
            loaded.append(np.load(file_name + '.npy'))
        else:
            loaded.append(None)
    return ifexists,loaded

def Save_Mat(file_name, matrix):
    np.save(file_name + '.npy', matrix)

def Save_Mats(file_names, matrices):
    for file_name, matrix in zip(file_names, matrices):
        np.save(file_name + '.npy', matrix)

def draw_mat(SOT, SOT_kmeans, EMD, Wass):
    msot = (SOT).cpu().detach().numpy()
    msotk = (SOT_kmeans).cpu().detach().numpy()
    memd = (EMD).cpu().detach().numpy()
    mwass = (Wass).cpu().detach().numpy()
    # msliced = (Sliced).cpu().detach().numpy()
    # mLOT = (LOT).cpu().detach().numpy()
    msot_emd = torch.abs((SOT) - (EMD)).cpu().detach().numpy()
    msotk_emd = torch.abs((SOT_kmeans) - (EMD)).cpu().detach().numpy()
    mwass_emd = torch.abs((Wass) - (EMD)).cpu().detach().numpy()
    # msliced_emd = torch.abs((Sliced) - (EMD)).cpu().detach().numpy()
    # mLOT_emd = torch.abs((LOT) - (EMD)).cpu().detach().numpy()
    vmax = torch.max(torch.FloatTensor([SOT.max(), SOT_kmeans.max(), EMD.max(), Wass.max()])).cpu().item()
    plt.subplot(2, 4, 1)
    plt.title(f'eSOT')
    plt.imshow(msot, vmin = 0, vmax = vmax)
    plt.colorbar()
    plt.subplot(2, 4, 2)
    plt.title('OT-EMD')
    plt.imshow(memd, vmin = 0, vmax = vmax)
    plt.colorbar()
    plt.subplot(2, 4, 3)
    plt.title('eOT-Sinkhorn')
    plt.imshow(mwass, vmin = 0, vmax = vmax)
    plt.colorbar()

    plt.subplot(2, 4, 4)
    plt.title('eSOT-k')
    plt.imshow(msotk, vmin = 0, vmax = vmax)
    plt.colorbar()
    # plt.subplot(3, 4, 5)
    # plt.title(f'Sliced Wasserstein distance')
    # plt.imshow(msliced, vmin = 0, vmax = vmax)
    # plt.colorbar()
    # plt.subplot(3, 4, 6)
    # plt.title(f'Low rank Sinkhorn')
    # plt.imshow(mLOT, vmin = 0, vmax = vmax)
    # plt.colorbar()
    plt.subplot(2, 4, 5)
    plt.title(f'Diff eSOT - OT-EMD')
    plt.imshow(msot_emd, vmin = 0, vmax = vmax)
    plt.colorbar()

    plt.subplot(2, 4, 6)
    plt.title(f'Diff SOT-k - OT-EMD')
    plt.imshow(msotk_emd, vmin = 0, vmax = vmax)
    plt.colorbar()
    plt.subplot(2, 4, 7)
    plt.title(f'Diff eOT - OT-EMD')
    plt.imshow(mwass_emd, vmin = 0, vmax = vmax)
    plt.colorbar()
    # plt.subplot(3, 4, 10)
    # plt.title(f'Diff Sliced - EMD')
    # plt.imshow(msliced_emd, vmin = 0, vmax = vmax)
    # plt.colorbar()
    # plt.subplot(3, 4, 11)
    # plt.title(f'Diff Low Rank Sinkhorn - EMD')
    # plt.imshow(mLOT_emd, vmin = 0, vmax = vmax)
    # plt.colorbar()
    plt.show()

def draw_mat2(SOT, EMD):
    msot = (SOT).cpu().detach().numpy()
    memd = (EMD).cpu().detach().numpy()
    msot_emd = torch.abs((SOT) - (EMD)).cpu().detach().numpy()
    vmax = torch.max(torch.FloatTensor([SOT.max(), EMD.max()])).cpu().item()
    plt.subplot(1, 3, 1)
    plt.title(f'SOT')
    plt.imshow(msot, vmin = 0, vmax = vmax)
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title('EMD')
    plt.imshow(memd, vmin = 0, vmax = vmax)
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title(f'Diff SOT - EMD')
    plt.imshow(msot_emd, vmin = 0, vmax = vmax)
    plt.colorbar()
    plt.show()

def draw_basis(basis):
    plt.title('Learned Basis')
    plt.imshow(basis)
    plt.colorbar()

def draw_curves_err(num_epochs : int, losses : dict, title = 'Average approximation loss', xlabel = "Epoch", ylabel = "Loss", xaxis = None, ylim = None):
    if xaxis is None:
        x = np.arange(num_epochs)
    else:
        x = [xaxis[_] for _ in range (num_epochs)]

    others = []

    for key, item in losses.items():
        if not isinstance(item[2], list):
            others.append((key, item[0], item[1], [item[2] for i in range(num_epochs)]))
        else:
            others.append((key, item[0], item[1], item[2]))

    for label, color, ls, curve in others:
        plt.plot(x, curve, color = color, ls = ls, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if not ylim is None:
        plt.ylim(ylim)

    plt.legend()

def draw_curves_err_marker(num_epochs : int, losses : dict, title = 'Average approximation loss', xlabel = "Epoch", ylabel = "Loss", xaxis = None, ylim = None):
    if xaxis is None:
        x = np.arange(num_epochs)
    else:
        x = [xaxis[_] for _ in range (num_epochs)]

    others = []

    for key, item in losses.items():
        if not isinstance(item[2], list):
            others.append((key, item[0], item[1], [item[2] for i in range(num_epochs)], item[3]))
        else:
            others.append((key, item[0], item[1], item[2], item[3]))

    for label, color, ls, curve, mk in others:
        plt.plot(x, curve, color = color, ls = ls, label=label, marker = mk)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if not ylim is None:
        plt.ylim(ylim)

    plt.legend()

def draw_hists_err(num_epochs : int, losses : dict, title = 'Average approximation loss', xlabel = "Epoch", ylabel = "Loss", xaxis = None, total_width = 10.0):
    if xaxis is None:
        x = np.arange(num_epochs)
    else:
        x = np.array([xaxis[_] for _ in range (num_epochs)])

    others = []

    total_width, n = 0.8, len(losses)
    width = total_width / n
    x = x - (total_width - width) / 2

    for key, item in losses.items():
        if not isinstance(item[2], list):
            others.append((key, item[0], item[1], [item[2] for i in range(num_epochs)]))
        else:
            others.append((key, item[0], item[1], item[2]))

    cidx = 0
    for label, color, ls, curve in others:
        plt.bar(x + cidx*width, curve, width = width, color = color, label=label)
        cidx = cidx + 1

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend()

def err_compute(A, B):
    #return np.abs(A - B).mean()
    return np.sqrt(np.mean(np.power(A - B,2)))