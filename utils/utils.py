import torch
import os
import numpy as np
from glob import glob
import openmesh as om


def makedirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_sparse(spmat):
    return torch.sparse.FloatTensor(
        torch.LongTensor([spmat.tocoo().row,
                          spmat.tocoo().col]),
        torch.FloatTensor(spmat.tocoo().data), torch.Size(spmat.tocoo().shape))


def to_edge_index(mat):
    return torch.LongTensor(np.vstack(mat.nonzero()))


def preprocess_spiral(face, edges, seq_length, vertices=None, dilation=1):
    from .generate_spiral_seq import extract_spirals
    assert face.shape[1] == 3
    if vertices is not None:
        mesh = om.TriMesh(np.array(vertices), np.array(face))
    else:
        n_vertices = face.max() + 1
        mesh = om.TriMesh(np.ones([n_vertices, 3]), np.array(face))
    spirals,indexs = extract_spirals(mesh, edges, seq_length=seq_length, dilation=dilation)
    spirals = torch.tensor(spirals)
    indexs = torch.tensor(indexs)
    return spirals,indexs