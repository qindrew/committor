import logging
import os
from glob import glob
from pathlib import Path
from typing import List, Union

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from jax import numpy as np
import numpy as onp
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from e3nn import o3

from mace import data
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric, torch_tools, utils
from mace.tools.compile import prepare
from mace.tools.scripts_utils import extract_model
from ase.io import read
from functools import partial
from tqdm import tqdm
import torch.nn as nn


class Committor_NN(nn.Module):
    """
    MLP for estimating the committor. The model wraps a frozen MACE model to compute atomic environment descriptors. 
    The MLP is applied atomwise, then summed to produce a scalar value which is then fed into a sigmoid-like function,
    bounding the output value between 0 and 1.

    Args:
        input_dim (int): Dimensionality of input features from the descriptor.
        h1 (int): Number of hidden units in the first MLP layer.
        h2 (int): Number of hidden units in the second MLP layer.
        h3 (int): Number of hidden units in the third MLP layer.
        output_dim (int): Output dimensionality (default is 1 for scalar committor).
        batch_size (int): Number of systems per batch.
        sig_k (float): Scaling factor for sigmoid-like activation.
        mace_path (str): Path to the pretrained MACE model file.
    """
    def __init__(self, input_dim=256, h1=64, h2=32, h3=16, output_dim=1, batch_size=5, device='cuda',
                 sig_k=3, mace_path='/lcrc/globalscratch/acqin2/mlip_md/committee3/mace2.model'):
        super().__init__()

        self.sig_k = sig_k
        mace_model = torch.load(mace_path, map_location=device)
        mace_model.to(device)
        for param in mace_model.parameters():
            param.requires_grad = False
        self.mace_model = mace_model

        self.z_table = utils.AtomicNumberTable([int(z) for z in mace_model.atomic_numbers])
        self.keyspec = data.KeySpecification(info_keys={}, arrays_keys={"charges": "Qs"})

        irreps_out = o3.Irreps(str(mace_model.products[0].linear.irreps_out))
        self.l_max = irreps_out.lmax
        self.num_invariant_features = irreps_out.dim // (irreps_out.lmax + 1) ** 2
        self.num_layers = mace_model.num_interactions

        self.atom_mlp = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, output_dim),
        ).to(device)

        self.batch_size = batch_size

    def sigmoid_like(self, x):
        return torch.sigmoid(self.sig_k * x)

    def forward(self, batch, training=False):
        batchdict = batch.to_dict()

        descriptor = self.mace_model(
            batchdict,
            training=training,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            compute_hessian=False,
            compute_edge_forces=False,
            compute_atomic_stresses=False,
        )["node_feats"]

        invariant = extract_invariant(
            descriptor,
            num_layers=self.num_layers,
            num_features=self.num_invariant_features,
            l_max=self.l_max,
        )  # [batchsize * Natoms, input_dim]

        atom_outputs = self.atom_mlp(invariant)  # [batchsize * Natoms, 1]
        batch_output = atom_outputs.reshape(self.batch_size, -1, 1).sum(dim=-2)  # [batchsize, 1]

        if training:
            return self.sigmoid_like(batch_output).squeeze(), batchdict
        else:
            return self.sigmoid_like(batch_output).squeeze()
        

class BoundaryLoss(nn.Module):
    """
    Custom loss function for committor neural networks incorporating two terms:
    
    1. Gradient loss: Encourages smoothness by penalizing large gradients of the committor 
       function with respect to atomic positions. Kolgomorov Loss
    2. Boundary loss: Applies a potential well-like penalty to match boundary conditions:
       - For label 0: committor = 0 (A-bound state)
       - For label 2: committor = 1 (B-bound state)

    Args:
        gradient_weight (float): Scaling factor for the gradient loss term.
        boundary_weight (float): Scaling factor for the boundary condition loss term.
        masses (torch.Tensor): Tensor of atomic masses used for mass-weighted gradients.
    """
    def __init__(self, gradient_weight=1.0, boundary_weight=1.0, masses=None):
        super().__init__()
        self.gradient_weight = gradient_weight
        self.boundary_weight = boundary_weight
        self.masses = masses

    def forward(self, model, batch):
        output, batchdict = model(batch, training=True)

        boundary_loss = torch.tensor(0.0, device=output.device)
        gradient_loss = torch.tensor(0.0, device=output.device)
        label = batchdict['label']
        batch_size = len(label)

        gradients = torch.autograd.grad(
            outputs=output.sum(),
            inputs=batchdict['positions'],
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.reshape(model.batch_size, -1, 3)  # [batch, atom, 3]

        gradient_loss += ((1 / self.masses) * gradients ** 2).sum(dim=(-2, -1)).mean()

        A_bound = output.pow(2)
        B_bound = (output - 1.0).pow(2)

        boundary_loss += torch.where(label == 0, A_bound, torch.where(label == 2, B_bound, 0)).mean()

        total_loss = self.gradient_weight * gradient_loss + self.boundary_weight * boundary_loss

        #TO DO. REWEIGHT BIAESD DATA TO UNBIASED
        return total_loss, gradient_loss, boundary_loss
    

def train_model(model, train_data_loader, val_data_loader, criterion, optimizer, scheduler, num_epochs=100, device='cuda'):
    """
    Train a Committor NN using the provided data loaders, loss function, optimizer, and scheduler.

    Args:
        model (nn.Module): The neural network model to train.
        train_data_loader (DataLoader): DataLoader for training data.
        val_data_loader (DataLoader): DataLoader for validation data.
        criterion (callable): Loss function returning (total_loss, grad_loss, bound_loss).
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs to train.
    """
    for epoch in range(num_epochs):
        total_loss = total_grad_loss = total_bound_loss = 0.0
        model.train()

        for batch in train_data_loader:
            optimizer.zero_grad()
            loss, grad_loss, bound_loss = criterion(model, batch.to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_grad_loss += grad_loss.item()
            total_bound_loss += bound_loss.item()

        print('Epoch =', epoch)
        print('average train loss:', total_loss / len(train_data_loader))
        print('average train grad loss:', total_grad_loss / len(train_data_loader))
        print('average train bound loss:', total_bound_loss / len(train_data_loader))

        total_loss = total_grad_loss = total_bound_loss = 0.0
        model.eval()

        for batch in val_data_loader:
            optimizer.zero_grad()
            loss, grad_loss, bound_loss = criterion(model, batch.to(device))

            total_loss += loss.item()
            total_grad_loss += grad_loss.item()
            total_bound_loss += bound_loss.item()

        print('average val loss:', total_loss / len(val_data_loader))
        print('average val grad loss:', total_grad_loss / len(val_data_loader))
        print('average val bound loss:', total_bound_loss / len(val_data_loader))
        print('\n')

        scheduler.step(total_loss)


def preprocess_data(xyz_path: str, mace_path: str, labels, device: str = 'cuda', batch_size: int = 5, 
                    val_frac: float = 0.5, seed: int = None, directory: str = '.', split: bool = True):
    """
    Preprocess simulation trajectory in xyz format for training/validation into
    inputs for a pretrained MACE model.

    Args:
        xyz_path (str): Path to the input `.xyz` or `.extxyz` trajectory file.
        mace_path (str): Path to the pretrained MACE model.
        labels (List[int]): List of boundary condition labels (0 or 2) for each structure.
        device (str): Torch device for model and data handling ('cpu' or 'cuda').
        batch_size (int): Batch size for the returned DataLoader(s).
        val_frac (float): Fraction of data to use for validation if `split=True`.
        seed (int): Random seed for data splitting.
        directory (str): Directory to cache split indices if needed.
        split (bool): Whether to split data into training and validation sets.

    Returns:
        If `split=True`: (train_data_loader, val_data_loader)
        Else: data_loader for the full dataset
    """
    atoms = read(xyz_path, format='extxyz', index=':')
    configs = data.utils.config_from_atoms_list(atoms, data.utils.KeySpecification())

    mace_model = torch.load(mace_path, map_location=device).to(device)
    z_table = utils.AtomicNumberTable([int(z) for z in mace_model.atomic_numbers])

    if split:
        total_dataset = [
            data.AtomicData.from_config(config, z_table=z_table, cutoff=4.)
            for config in configs
        ]
        for i, config in enumerate(total_dataset):
            config.label = labels[i]

        train_set, val_set = data.utils.random_train_valid_split(
            total_dataset, val_frac, seed, directory
        )

        train_data_loader = torch_geometric.dataloader.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_data_loader = torch_geometric.dataloader.DataLoader(
            dataset=val_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        return train_data_loader, val_data_loader

    else:
        total_dataset = [
            data.AtomicData.from_config(config, z_table=z_table, cutoff=4.)
            for config in configs
        ]
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=total_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        return data_loader


if __name__ == '__main__':

    '''
    Example code to train Committor NN model. Also includes example of labeling configurations into metastable states based on order parameter thresholds. 
    '''
    def coordnum_exp_no_overlap(r, r0s=None, ks=None, locations=None, coeffs=None, cell_size=None):
        rO7 = r[0]
        rO = r[:4]
        rH = r[4:]

        vals = switch_exp(np.linalg.norm(rO[:,None,:] - rH, axis=2), r0=r0s, k=ks) #(4,4)
        maxval = np.max(vals, axis=0)
        filtered = np.where(vals < maxval, 0.0, vals) #coordination number of each O
        filtered2 = filtered.sum(axis=1)
        maxvalids = np.argsort(filtered2)[-2:]
        waters = rO[maxvalids] #(2,3) two maximally coordinated Os
        minOindex = np.where(filtered2[maxvalids[0]] > 1.92, maxvalids[np.argmin(np.linalg.norm(rO7-waters,axis=1))], maxvalids[1])
        maxHindex = np.argmax(np.where(filtered[minOindex]>0,vals[0],0)) #of minO Hs, index of H closest to O7
        return np.where(np.max(filtered[0])>0,
                        vals[minOindex,np.argmax(filtered[0])] - np.max(filtered[0]),
                        filtered[minOindex,maxHindex] - vals[0,maxHindex])
    
    def distance(r, cell_size=None):
        diff = r[1:] - r[0]
        #diff = diff - np.round(diff / cell_size) * cell_size
        return np.linalg.norm(diff,axis=1)

    def calc_coordination_exp(r, r0=None, k=None, cell_size=None):
        return switch_exp(distance(r,cell_size=cell_size), r0=r0, k=k).sum()

    def switch_exp(arr, r0=None, k=None,):
        return (1+np.exp(k*(arr-r0)))**-1
    
    atoms = read('train.xyz',format='extxyz',index=':')
    mace_path = '/lcrc/globalscratch/acqin2/mlip_md/committee3/mace2.model'
    device='cuda'
    configs = data.utils.config_from_atoms_list(atoms, data.utils.KeySpecification())
    mace_model = torch.load(mace_path,map_location=device).to(device)
    z_table = utils.AtomicNumberTable([int(z) for z in mace_model.atomic_numbers])
    total_dataset =[data.AtomicData.from_config(config, z_table=z_table, cutoff=4.)
            for config in configs
        ]
    arr1 = np.asarray([5,6,7,44,47])
    arr2 = np.asarray([7,6,44,47, 45,46,48,49,])
    r1s = []
    r2s = []
    for i in range(len(atoms)):
        r1s.append(np.asarray(atoms[i].get_positions()[arr1]))
        r2s.append(np.asarray(atoms[i].get_positions()[arr2]))
    r1s = np.stack(r1s,axis=0)
    r2s = np.stack(r2s,axis=0)
    from jax import jit, vmap
    CV1 = partial(calc_coordination_exp,r0=2.1,k=6.7)
    CV2 = partial(coordnum_exp_no_overlap, r0s=1.6, ks=8)
    CV1_vmapjit = jit(vmap(CV1,in_axes=0,out_axes=0))
    CV2_vmapjit = jit(vmap(CV2,in_axes=0,out_axes=0))
    CV1s = CV1_vmapjit(r1s)
    CV2s = CV2_vmapjit(r2s)
    stateA = np.where((CV1s < 2.5) & (CV2s > 0.3))[0]
    stateB = np.where((CV1s < 2.5) & (CV2s < -0.3))[0]
    labels = (np.zeros(len(CV2s)) + 1).astype(np.int32)
    labels = labels.at[stateA].set(0)
    labels = labels.at[stateB].set(2)
    labels = torch.tensor(onp.asarray(labels))
    masses = torch.tensor(atoms[0].get_masses()).to(torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    del mace_model

    torch.manual_seed(98371)
    xyz_path = 'train.xyz'
    mace_path = '/lcrc/globalscratch/acqin2/mlip_md/committee3/mace2.model'
    train_data_loader, val_data_loader = preprocess_data(xyz_path, mace_path, labels, val_frac = 0.3, device=device,batch_size=10)
    model = Committor_NN(batch_size=10).to(device)   # model = torch.load('model.pt') #
    criterion = BoundaryLoss(masses=masses,boundary_weight=10.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)
    train_model(model, train_data_loader, val_data_loader, criterion, optimizer, scheduler, num_epochs=30)
    torch.save(model,'model.pt')
