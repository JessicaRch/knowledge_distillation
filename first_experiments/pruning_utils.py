import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import re
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
import cupy as cp
import h5py
from DSU import getComponents

# ----------------------------
# MACROS
# ----------------------------
e_ortho = 0.01 # value for consider two vectors epsilon-orthogonal (for pruning)

# ----------------------------
# Gradient & neuron utility functions
# ----------------------------

def collect_prunable_layer_gradients(results, max_layers=None):
    layers_name = [key for key in results[0]['grads'].keys() if 'weight' in key]
    sorted_layers_name = sorted(
        layers_name,
        key=lambda s: int(re.search(r"\d+", s).group())
    )
    pred_label = torch.cat([res['pred_labels'] for res in results], dim=0)

    layer_grads = {}
    for layer_name in sorted_layers_name[:-1]:
        grads = torch.cat([res['grads'][layer_name] for res in results], dim=0)
        layer_grads[layer_name] = grads

    return sorted_layers_name, pred_label, layer_grads

def sort_indexes(pred_labels):
    return torch.argsort(pred_labels, descending=False)

def get_indexes_diff(labels):
    labels = labels.squeeze()
    change_idx = torch.where(labels[1:] != labels[:-1])[0] + 1
    
    return torch.cat([
        torch.tensor([0], device=labels.device),
        change_idx,
        torch.tensor([len(labels)], device=labels.device)
    ])

def compute_similarity_matrix(grads_sorted):
    """
    grads_sorted: Tensor (N_samples, N_neurons, D)
    Returns similarity matrix: (N_neurons, N_samples, N_samples)
    """
    G = grads_sorted.permute(1, 0, 2)
    normG = torch.norm(G, dim=2, keepdim=True)
    GT = grads_sorted.permute(1, 2, 0)
    GGT = torch.bmm(G, GT)
    norm = torch.bmm(normG, normG.permute(0,2,1))
    Matrix = torch.zeros_like(GGT)
    mask = norm != 0
    Matrix[mask] = GGT[mask] / norm[mask]
    return Matrix

def cos_sim_mean(cos_sim, labels_pos):
    nneurons = cos_sim.shape[0]
    total_labels = len(labels_pos)-1
    mean_sim = torch.zeros((nneurons, total_labels, total_labels))
    for j in range(total_labels):
        for k in range(total_labels):
            start_j, end_j = labels_pos[j], labels_pos[j+1]
            start_k, end_k = labels_pos[k], labels_pos[k+1]
            mean_sim[:,j,k] = cos_sim[:, start_j:end_j, start_k:end_k].mean(axis=(1,2))
    return mean_sim


def conv_cosine_similarity(grad, labels_pos, threshold=0.09):

    output_dim = len(labels_pos) - 1
    if len(grad.shape) > 3:
        nsamples, nneurons, c, h, w = grad.shape
        grad = grad.view(nsamples, nneurons, c*h*w)          
    else:
        nsamples, nneurons, features = grad.shape
  
    norm_grad = F.normalize(grad, dim=-1)     # normaliza na dim 9
    # reorganiza pra facilitar o matmul

    permuted_grad = norm_grad.permute(1, 0, 2)          

    # similaridade do cosseno
    cos = torch.matmul(permuted_grad, permuted_grad.transpose(-1, -2))
    
    nneurons, h, w = cos.shape
    low_activity_neurons = []
    for i in range(nneurons):
        nonzero_frac = torch.count_nonzero(cos[i]) /  (h * w)
        if nonzero_frac < threshold:
            low_activity_neurons.append(i)
    
    return cos_sim_mean(cos, labels_pos), low_activity_neurons


def cosine_block_mean(m1, m2, device='cuda'):

    if not isinstance(m1, torch.Tensor):
        m1 = torch.tensor(m1)
    if not isinstance(m2, torch.Tensor):
        m2 = torch.tensor(m2)
    
    m1 = m1.to(device)
    m2 = m2.to(device)

    if m1.ndim == 2:
        norm1 = torch.linalg.norm(m1, dim=1, keepdim=True)
        norm2 = torch.linalg.norm(m2, dim=1, keepdim=True)
        norm  = norm1 @ norm2.T
        dot   = m1 @ m2.T
        axis  = (0, 1)
    else:
        norm1 = torch.linalg.norm(m1, dim=2, keepdim=True)
        norm2 = torch.linalg.norm(m2, dim=2, keepdim=True)
        norm2 = norm2.permute(1, 2, 0)
        norm1 = norm1.permute(1, 0, 2)
        norm  = norm1 @ norm2
        m2    = m2.permute(1, 2, 0)
        m1    = m1.permute(1, 0, 2)
        dot   = m1 @ m2
        axis  = (1, 2)

    div = torch.where(norm == 0, torch.zeros_like(dot), dot / norm)
    
    result         = div.mean(dim=axis).cpu()
    count_nonzero  = (div.abs() > e_ortho).float().mean(dim=axis).cpu()

    return count_nonzero.numpy(), result.numpy()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_nneurons_in_file(filename, layer_name):
    with h5py.File(filename, 'r') as f:
        first_group = list(f.keys())[0]
        nneuron = f[first_group]['grads'][layer_name].shape[1]
        return nneuron
        
def get_grads_in_file(filename, group, layer_name, neuron_index):
    with h5py.File(filename, 'r') as f:
        first_group = list(f.keys())[group]
        return  f[first_group]['grads'][layer_name][:,neuron_index,:][:]

def compute_mean_cos_cuda_sliced(sorted_grads, labels_pos, layers, device="cuda", filename=None):

    mean_sim = {}
    nlabels = len(labels_pos) - 1
    activation_per_layer = {}
    for layer_name in layers:
        if filename is not None:
            N_neurons = get_nneurons_in_file(filename, layer_name)
        else:
            if type(sorted_grads) != dict:
                grads = torch.cat([res['grads'][layer_name] for res in sorted_grads], dim=0)  # (N_samples, N_neurons, D)
            else:
                grads = sorted_grads[layer_name]  # (N_samples, N_neurons, D)
            grads = grads.view(grads.size(0), grads.size(1), -1)
            N_neurons = grads.shape[1]

        mean_sim_layer = torch.zeros(
            (N_neurons, nlabels, nlabels),
            device="cpu"
        )
        activation_per_layer[layer_name] = {}
        for n in range(N_neurons):
            # ---- move ONE neuron to GPU ----
                
                
            activation = 0
            p = 0
            for l1 in range(nlabels):
                if filename is None:
                    s1 = slice(labels_pos[l1], labels_pos[l1+1])
                    G1 = grads[s1.start:s1.stop, n, :].to(device)
                else:
                    grads = get_grads_in_file(filename, l1, layer_name, n)  # (N_samples, D)
                    grads = torch.tensor(grads).to(device)
                    G1 = grads.to(device)
                G1 = G1 - G1.mean(dim=0, keepdim=True)  # centraliza os gradientes
                # diagonal block
                act, mean_label = cosine_block_mean(G1, G1,device=device)
                mean_sim_layer[n, l1, l1] = torch.tensor(mean_label)
                activation += act
                p+=1
                for l2 in range(l1 + 1, nlabels):
                    s2 = slice(labels_pos[l2], labels_pos[l2+1])
                    if filename is None:
                        G2 = grads[s2.start:s2.stop, n, :].to(device)
                    else:
                        grads = get_grads_in_file(filename, l2,layer_name, n)  # (N_samples, D)
                        grads = torch.tensor(grads).to(device)
                        G2 = grads.to(device)
                    G2 = G2 - G2.mean(dim=0, keepdim=True)  # centraliza os gradientes
                    act, mean_label = cosine_block_mean(G1, G2, device=device)
                    mean_sim_layer[n, l1, l2] = torch.tensor(mean_label)
                    mean_sim_layer[n, l2, l1] = torch.tensor(mean_label)
                    activation += act 
                    p+=1
            activation_per_layer[layer_name][n] = activation/p
            # optional: free memory aggressively
            try:
                del G1, G2
            except NameError:
                pass
            torch.cuda.empty_cache()
        mean_sim[layer_name] = mean_sim_layer

    return mean_sim, activation_per_layer

def low_activation_grads(grads, threshold=0.1):
    active = grads.abs() > 0
    activity_ratio = active.float().mean(dim=(0,2))
    low_act_indices = torch.where(activity_ratio < threshold)[0]
    return low_act_indices

def get_grads_per_layer(filename):
    '''
        this function 
    '''

    with h5py.File(filename, 'r') as f:
        layers_name = sorted(f['prediction_0/grads'].keys(), key=sort_key)
        grads = {layer: f['prediction_0/grads'][layer][:] for layer in layers_name}
    return grads, layers_name

def get_grads(results, get_pred=False, only_pos=True):
 
    # --- labels e posições (leve, sem gradientes) ---
    pred_labels = torch.cat([res['pred_labels'] for res in results], dim=0)
    idx = sort_indexes(pred_labels)
    sorted_pred_labels = pred_labels[idx]
    pos = get_indexes_diff(sorted_pred_labels)
    
    layers_name = sorted(
        [k for k in results[0]['grads'] if 'weight' in k],
        key=sort_key
    )
    if only_pos: 
        return pos, layers_name
    # processa uma camada por vez para não acumular tudo na RAM
    print('inicio da coleta de gradientes por camada')
    grads = {}
    for layer in layers_name[:-1]:
        print(layer)
        # concatena só esta camada, reordena, e libera as referências intermediárias
        intermediario = np.concatenate([res['grads'][layer] for res in results], axis=0)
        intermediario = intermediario[idx]  # reordena os gradientes conforme as labels ordenadas
        print(f"shape dos gradientes coletados para {layer}: {intermediario.shape}")
        grads[layer] = torch.tensor(intermediario)

    if get_pred:
        return grads, pos, layers_name, sorted_pred_labels
    print('finalizou a coleta de gradientes por camada')
    return grads, pos, layers_name


def get_parameters_to_prune(grads, pos, layers, low_act_threshold=0.09, sim_threshold=0.05, filename=None):
    
    mean_cossim_per_layer = {}
    low_activity_per_layer = {}
    mean_cossim_per_layer, activation_per_layer = compute_mean_cos_cuda_sliced(grads, pos,layers, filename=filename)
    low_activity_per_layer = {}
    for layer in activation_per_layer:
        low_activity_per_layer[layer] = set()
        for n, act in activation_per_layer[layer].items():
            if act < low_act_threshold:
                low_activity_per_layer[layer].add(n)   

    rank = similarity_rank(mean_cossim_per_layer)

    sim_activity_per_layer = chose_neurons_to_prune(rank, sim_threshold)
    return sim_activity_per_layer, low_activity_per_layer

   
def similarity_rank(mean_cossim):
    main_rank = {}
    for layer_name in mean_cossim:

        mean_sim = mean_cossim[layer_name].reshape(mean_cossim[layer_name].shape[0], -1)
        nneurons = mean_sim.shape[0]

        n_pairs = nneurons * (nneurons - 1) // 2    # integer number of unique pairs

        # columns: [neuron1, neuron2, similarity]
        rank = torch.zeros((n_pairs, 3), dtype=float)

        sim_scores = []
        j = 0
        for n1 in range(nneurons-1):
            for n2 in range(n1+1,nneurons):
                rank[j, 0] = n1
                rank[j, 1] = n2
                rank[j, 2] = torch.cosine_similarity(mean_sim[n1], mean_sim[n2], dim=0)
                j+=1
        # Sort neurons by similarity score (ascending)
        sorted_neurons = sorted(rank, key=lambda x: x[2])
        main_rank[layer_name] = sorted_neurons

    return main_rank

def create_adjacency_list(ranked_neurons, threshold=0.85):
    c = len(ranked_neurons)*2
    V = int((1 + np.sqrt(1 + 4*c)) // 2)  # upper bound on number of neurons

    adj = [[] for _ in range(V)]
    for n1, n2, sim in ranked_neurons:
        if sim > threshold:
            adj[int(n1)].append(int(n2))
            adj[int(n2)].append(int(n1))
    return adj

def chose_neurons_to_prune(similarity_rank_dict, threshold=0.05):
    neurons_to_prune = {}

    for layer_name, ranked_neurons in similarity_rank_dict.items():
        c = len(ranked_neurons)*2
        V = int((1 + np.sqrt(1 + 4*c)) // 2)  # upper bound on number of neurons
        
        adj = [[] for _ in range(V)]
        for n1, n2, sim in ranked_neurons:
            if sim > threshold:
                adj[int(n1)].append(int(n2))
                adj[int(n2)].append(int(n1))
        components = getComponents(adj)
        # print(allneurons)
        prune_set = set()
        for comp in components:
            if len(comp) <= 1:
                continue
            # prune all but one neuron in the component
            prune_set.update(comp[1:])
        neurons_to_prune[layer_name] = prune_set
    return neurons_to_prune
import copy

def get_remaining_neurons(parameters2prune, config):
    
    remaining_neurons_per_layers = defaultdict(dict)
    new_config = copy.deepcopy(config)

    for name in parameters2prune.keys():
        if 'conv' in name:
            conv = name.split('.')[1]
            num = int(conv.replace('conv','')) - 1  # conv1 -> 1
            layer = 'conv'      
            min_per_layer = 'min_neurons_per_conv_layer'
        elif 'fc' in name:
            fc = name.split('.')[1]
            num = int(fc.replace('fc','')) - 1  # fc1 -> 1
            layer = 'fc'
            min_per_layer = 'min_neurons_per_fc_layer'
        else:
            continue

        all_params = set(list(range(config[layer][num])))
        final =  all_params - parameters2prune[name]
        if len(final) < config[min_per_layer][num]:
            final = set(list(all_params)[:config[min_per_layer][num]])

        remaining_neurons_per_layers[name] = final
        new_config[layer][num] = len(final)

    return remaining_neurons_per_layers, new_config

def prune_model(model, pruned_model, remaining_neurons_layers_set, sorted_layers_name, config):
    
    prev_keep_idx = list(range(config['input_dim']))
    for i, layer_name in enumerate(sorted_layers_name[:-1]):
        keep_idx = list(remaining_neurons_layers_set[layer_name])
        idx = layer_name.split('.')[1]
        next_idx = sorted_layers_name[i+1].split('.')[1]
        
        next_keep_idx = list(remaining_neurons_layers_set[sorted_layers_name[i+1]])
        with torch.no_grad():
            # fc1: keep selected neurons (rows)
            pruned_model.layers[idx].weight.copy_(model.layers[idx].weight[keep_idx,:][:, prev_keep_idx])
            pruned_model.layers[idx].bias.copy_(model.layers[idx].bias[keep_idx])

            # fc2: keep corresponding input columns
            pruned_model.layers[next_idx].weight.copy_(model.layers[next_idx].weight[:,keep_idx][next_keep_idx,:])
            pruned_model.layers[next_idx].bias.copy_(model.layers[next_idx].bias[next_keep_idx])
        prev_keep_idx = keep_idx

def sort_key(name):
    m = re.search(r'\.(conv|fc)(\d+)\.weight$', name)
    if m is None:
        m = re.search(r'\.(conv|fc)(\d+)$', name)
        if m is None:
            raise ValueError(f"Unrecognized layer name: {name}")
    layer_type, idx = m.group(1), int(m.group(2))
    type_order = {'conv': 0, 'fc': 1}
    return (type_order[layer_type], idx)


def get_layers_name(model):
    layers_name = []
    for name, module in model.named_modules():
        if name.startswith('layers.conv') or name.startswith('layers.fc'):
            layers_name.append(name)

    layers_name = sorted(layers_name, key=sort_key)
    return layers_name

def get_feature_map_size(model, input_shape):
    device = next(model.parameters()).device  # pega device do modelo
    x = torch.zeros(1, *input_shape).to(device)  # move tensor para o mesmo device

    with torch.no_grad():
        for layer in model.layers.children():
            if isinstance(layer, nn.Flatten):
                break
            x = layer(x)
    _, C, H, W = x.shape
    return C, H, W


import torch
import h5py

def append_to_group(h5_group, data_dict):
    for key, value in data_dict.items():

        # If nested dictionary → recurse
        if isinstance(value, dict):
            subgroup = h5_group.require_group(key)
            append_to_group(subgroup, value)

        else:
            # Convert torch tensor to numpy
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()

            # Create dataset if it doesn't exist
            if key not in h5_group:
                maxshape = (None, *value.shape[1:])
                h5_group.create_dataset(
                    key,
                    data=value,
                    maxshape=maxshape,
                    chunks=True
                )
            else:
                dset = h5_group[key]
                old_size = dset.shape[0]
                new_size = old_size + value.shape[0]

                dset.resize((new_size, *value.shape[1:]))
                dset[old_size:new_size] = value

def save_results_to_h5(batch_data, filename, pos):

    with h5py.File(filename, 'a') as f:
        
        i = len(pos)-2 # classe 0, len(pos) = 1, classe 1, len(pos) = 2, ...
        class_group_name = f"prediction_{i}"
        if class_group_name not in f:
            f.create_group(class_group_name)
    
        # ------------------------------------------------
        start, end = pos[i], pos[i+1]

        class_batch_data = {
            "grads": {name: g for name, g in batch_data["grads"].items()}
        }

        # cria grupo grads se não existir
        grads_group = f[class_group_name].require_group("grads")
        for name, g in class_batch_data["grads"].items():
                # Se dataset ainda não existe → cria
            if name not in grads_group:
                grads_group.create_dataset(
                    name,
                    data=g,
                    maxshape=(None, *g.shape[1:]),  # permite crescer no eixo 0
                    chunks=True
                )

            # Se já existe → faz append
            else:
                dset = grads_group[name]
                old_size = dset.shape[0]
                new_size = old_size + g.shape[0]

                dset.resize((new_size, *g.shape[1:]))
                dset[old_size:new_size] = g

if __name__ == "__main__":
    # main.py
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from pruning_utils import low_activity_neurons, create_adjacency_list, chose_neurons_to_prune

    from mnist_dataset import MNISTDataset

    from metrics import Evaluator
    from pruner import IterativePruner, Pruner
    import torch
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from gradients import GradMonitor
    from DSU import getComponents

    from model import CNN
    from training import Trainer
    from digits_dataset import DigitsDataset

    digits_dataset = DigitsDataset(cnn=True)
    train_loader, test_loader = digits_dataset.get_loaders(batch_size=32)

    x,y = next(iter(train_loader))
    input_dim = x.shape[2:]
    output_dim = 10

    config = {
        'input_dim': input_dim,  # (height, width)
        'conv': [16, 32],  
        'fc' : [64],
        'conv_kernel' : [3,3, 3], 
        'conv_stride': [1, 1, 1],
        'conv_padding': [1, 1, 1],
        'min_neurons_per_conv_layer': [4, 4],
        'min_neurons_per_fc_layer': [4],
        'max_pool_kernel' : [2,2],
        'max_pool_stride' : [2,2],
        'output_dim': output_dim,
        'criterion': torch.nn.CrossEntropyLoss(),
        'optimizer': torch.optim.Adam,
        'lr': 0.001
    }

    cnn_model = CNN(config)
    trainer = Trainer(cnn_model, config)
    for _ in range(20):
        trainer.train_epoch(train_loader)

    evaluator = Evaluator(cnn_model)
    train_acc, test_acc = evaluator.accuracy(train_loader), evaluator.accuracy(test_loader)
    grad_monitor = GradMonitor(cnn_model, config)
    print("Initial train acc:", train_acc, "test acc:", test_acc)

    pruner = IterativePruner(cnn_model, config, train_loader, test_loader, max_iters=2,
                            threshold_low_act=0.09, threshold_sim=0.8, use_cuda=False, eps=0.15)
    pruner.run_conv(retrain_epochs=10)