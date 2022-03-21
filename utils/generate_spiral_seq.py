import openmesh as om
from sklearn.neighbors import KDTree
import numpy as np
import networkx as nx

def _next_ring(mesh, last_ring, other,index):
    res = []
    res_index = []
    def is_new_vertex(idx):
        return (idx not in last_ring and idx not in other and idx not in res)

    for vh1 in last_ring:
        vh1 = om.VertexHandle(vh1)
        after_last_ring = False
        for vh2 in mesh.vv(vh1):
            if after_last_ring:
                if is_new_vertex(vh2.idx()):
                    res.append(vh2.idx())
                    res_index.append(index)

            if vh2.idx() in last_ring:
                after_last_ring = True
        for vh2 in mesh.vv(vh1):
            if vh2.idx() in last_ring:
                break
            if is_new_vertex(vh2.idx()):
                res.append(vh2.idx())
                res_index.append(index)
    return res,res_index


def extract_spirals(mesh, edges, seq_length, dilation=1):
    # output: spirals.size() = [N, seq_length]
    spirals = []
    indexs= []
    kd_tree = 0
    graph = nx.Graph()
    for edge in edges.T:
        edge = tuple(edge)
        graph.add_edge(*edge)

    for vh0 in mesh.vertices():
        i=0
        reference_one_ring = []
        reference_one_ring_index = []
        for vh1 in mesh.vv(vh0):
            reference_one_ring.append(vh1.idx())
            reference_one_ring_index.append(i+1)
        spiral = [vh0.idx()]
        spiral_index = [0]
        one_ring = list(reference_one_ring)
        one_ring_index = reference_one_ring_index
        #print('one ring',len(one_ring),len(one_ring_index))
        last_ring = one_ring
        last_ring_index = one_ring_index
        i = i+2
        next_ring,next_ring_index = _next_ring(mesh, last_ring, spiral, index=i)
        spiral.extend(last_ring)
        spiral_index.extend(last_ring_index)

        while len(spiral) + len(next_ring) < seq_length * dilation:
            if len(next_ring) == 0:
                break
            last_ring = next_ring
            last_ring_index = next_ring_index
            next_ring, next_ring_index = _next_ring(mesh, last_ring, spiral,i+1)
            i = i+1
            spiral.extend(last_ring)
            spiral_index.extend(last_ring_index)
        if len(next_ring) > 0:
            spiral.extend(next_ring)
            spiral_index.extend(next_ring_index)
        else:
            #print('kD_TREE')
            kd_tree +=1
            kdt = KDTree(mesh.points(), metric='euclidean')
            spiral = kdt.query(np.expand_dims(mesh.points()[spiral[0]],
                                              axis=0),
                               k=seq_length * dilation,
                               return_distance=False).tolist()
            spiral = [item for subspiral in spiral for item in subspiral]
            spiral_index = get_edge_num_dis(graph,spiral)


        spirals.append(spiral[:seq_length * dilation][::dilation])
        indexs.append(spiral_index[:seq_length * dilation][::dilation])
    print(kd_tree)
    return spirals,indexs

def get_edge_num_dis(g,sprial_list):
    source_node = sprial_list[0]
    dis_list = [0]
    for node in range(1,len(sprial_list)):
        try:
            path = nx.shortest_path(g,source_node,sprial_list[node])
            dis = len(path)
        except nx.NetworkXNoPath:
            dis = -1
            #print('no path')
        dis_list.append(dis)
    return dis_list