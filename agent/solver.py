import math
from functools import reduce
from itertools import chain, product

import numpy as np
import torch
from scipy.linalg import block_diag
from collections import defaultdict
import heapq
from rules.rule import hash
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm
import random
from torch_geometric.data import Data, DataLoader

class Solver:
    def __init__(self, optimizer):
        # optimizer
        self.optimizer = optimizer

    def __call__(self, env, dispatch, kwargs):
        # iteratively process
        experience, num_assigned = [], 0.0
        #  constraint
        constraints = Constraints()
        # renew group searcher
        searcher = GroupSearcher()
        while True:
            # the group to be optimizer 1 XS
            group = searcher(dispatch, env, kwargs)
            # non-feasible in the group or reach the time
            # or (time.perf_counter() - _start >= kwargs['force'])
            if group is None: break
            # feasible zone F1 x S, edges F1*S1 x F1*S1
            feasible, graph, crafts, sites = constraints(env, dispatch, group)
            # print(sites)
            # print(crafts)
            # optimizer
            assignments,data = self.optimizer(dispatch, feasible, graph, crafts, sites)
            # print(sites)
            # print(assignments)
            # empty
            if not len(assignments):
                dispatch.feasible[0, :, sites] = False
                continue
            # feasible craft and previously preference
            # crafts, ranking = crafts[feasible[:, sites[site]]], []
            # ranking order of the assignment
            ranking = []
            count = 0
            # iteratively to assign the crafts

            for craft, site in assignments:
                # whether the allocation is legal:
                if not dispatch.feasible[0, craft, site]:
                    continue
                    # assign
                env.step(dispatch, craft, site)
                # real ranking
                ranking.append((craft, site))
           # dispatch.feasible[0, :, sites] = False

            if kwargs['train']:
                # reward: number of feasible
                # r = len(ranking) / (dispatch.assignments.shape[-1] - 1.0)
                r = dispatch.soft
                experience.append(data)

        return experience



class GroupSearcher:
    def __init__(self):
        # priorities on the group of the sits
        self.groups = None

    def group(self, env, dispatch):
        # taxi/in/out correlation 2  X h  x S
        data = env.constraints[2].rules[0].criteria(dispatch.flags, None, None)[0]
        # S x h  @ h x S -> S x S
        data = data[0].T @ data[1]
        # (a,b), (b,e) -> (a,b,e)
        data = np.logical_or(data, data.T)
        # correlation with itself
        np.fill_diagonal(data, True)
        # mask the correlation among other sites with None position
        data[1:, 0], data[0, :] = False, False
        # symmetric
        # 2^n, 2^ n -> 2 ^ (n+1)
        for _ in range(math.ceil(math.log2(data.shape[-1]))): data = data @ data
        # (a,b), (b,c) -> C x S
        return hash(data[np.newaxis, ...])[0]

    def __call__(self, dispatch, env, kwargs):
        # group the positions
        if self.groups is None:
            # taxi group 2 hops
            groups = self.group(env, dispatch)
            # the main objective
            objective = env.eval.object.astype(np.bool_)
            # priorities
            priorities = (groups & objective, groups & (~objective))
            # a group of sites
            self.idx = (np.arange(objective.size)[objective], np.arange(objective.size)[~objective])
            # filter the empty for groups,  (2, C x S) & S -> (2, C x S)
            self.groups = list(map(lambda t: t[np.any(t, axis=-1)], priorities))
            # focus on the target group
            self.choice = 0

        #  C XS, (F-k) x S  ->C1 x S1, (F-k) X S1,
        components = list(map(lambda t: (t[0][:, t[1]], t[-1][:, t[1]]),
                              list(zip(self.groups, self.idx, [dispatch.feasible[~dispatch.flags]] * len(self.idx)))))

        groups, feasible = components[self.choice]
        #  (F-k) x S1,  C x S1 -> (F-k) X C
        zone = feasible @ groups.T
        # over
        swap = ~np.any(zone)
        # terminates
        if swap and (self.choice == 1): return None
        # turns to the next group
        if swap: self.choice = 1
        # re-calculate the zone
        if swap: zone = components[self.choice][1] @ components[self.choice][0].T
        #  C x S
        groups = self.groups[self.choice]
        # normalized the importance of craft w.r.t to its feasible cluster
        # (F-k) X C
        zone = zone / (np.sum(zone, axis=-1, keepdims=True) + 1e-5)
        #  number of feasible crafts on the groups
        # (F-k) x C -> 1 x C
        choice = np.sum(zone, axis=0)
        # filter the group with zero flights
        groups, choice = groups[choice > 0], choice[choice > 0]
        # empty
        if not len(groups): return None
        # less feasible with less source
        g = np.lexsort((np.sum(groups, axis=-1), choice))[0]
        #  select the group with the minimal feasible solutions
        return groups[g]

    # output the conflict
    def hook(self, env, dispatch):
        # unzip the groups
        groups = list(chain(*[self.groups[0], self.groups[1]]))
        # position raw
        positions = env.data.positions.fields['position_no'].raw[0]
        # craft raw
        grants = env.data.plannings.fields['grant_id'].raw[0]
        # in_position
        in_position = env.data.plannings.fields['in_position_time'].raw[0]
        out_position = env.data.plannings.fields['out_position_time'].raw[0]
        # enumerate all the groups
        for g in groups:
            # position groups
            sites = np.arange(g.size)[g]
            # overlaps F x S1, Fx 1 -> F x S1
            overlaps = (dispatch.positions.reshape(-1, 1) == sites) & (dispatch.assignments.reshape(-1, 1) == 0)
            # non overlaps
            if not np.any(overlaps): continue
            # iter
            for site in sites:
                # filter the crafts
                same = (dispatch.assignments.reshape(-1) == site) & (dispatch.positions.reshape(-1) == site)
                # diff
                diff = (dispatch.assignments.reshape(-1) == 0) & (dispatch.positions.reshape(-1) == site)
                # format
                format = lambda flags: [(grants[k], in_position[k], out_position[k]) for k, v in enumerate(flags) if v]
                # same -> diff
                print(f'{[positions[k] for k in sites]}\t{positions[site]}\t{format(same)}\t->\t{format(diff)}')


class Constraints:
    def __init__(self):
        pass

    def __call__(self, env, dispatch, group):
        # related sites S2,
        group_sites = np.arange(group.size)[group]

        # feasible sites B x F x S2 -> S1,
        sites = group_sites[np.any(dispatch.feasible[~dispatch.flags][:, group_sites], axis=0)]

        # build the feasible adjacent B x F x S1 -> F x S1
        feasible = dispatch.feasible[..., sites].reshape(-1, sites.size)

        # cut off the crafts Fx S1 -> F
        masks = np.any(feasible, axis=-1)

        # crafts F1,
        crafts = np.arange(masks.size)[masks]

        # conflict on the pairs of (craft, site) F1*S1
        feasible = feasible[crafts].reshape(-1)

        # taxi feasible 2, F1*S1 x F1*S1 -> F1*S1 x F1*S1
        taxi = reduce(lambda x, y: x | y, self.correlate(env.constraints[2].rules[0], dispatch, crafts, sites))

        # overlaps F1*S1 x F1*S1
        laps = reduce(lambda x, y: x | y, self.correlate(env.constraints[2].rules[1], dispatch, crafts, sites, True))

        # only a position for a craft F1*S1 x F1*S1
        reflective = block_diag(*[np.ones((sites.size, sites.size), dtype=np.bool_)] * crafts.size)

        # Combine edges into a single constraint
        constraint = reduce(lambda x, y: x | y, [taxi, laps, reflective])

        # preference on the cluster, without of preference
        pref = np.ones_like(feasible) if len(env.constraints) < 4 else \
            reduce(lambda x, y: x + y, self.pref(env.constraints[-1].rules[0], dispatch, crafts, sites))

        pref = np.reshape(pref, -1) * feasible

        # Normalize pref to range [0, 5] with mean adjusted to 3.5
        if np.any(feasible):
            valid_pref = pref[feasible > 0]
            min_val = valid_pref.min()
            max_val = valid_pref.max()
            if max_val > min_val:
                # Normalize to 0-1
                normed = (valid_pref - min_val) / (max_val - min_val)

                # Scale to 0-5
                scaled = normed * 5

                # Adjust mean to 3.5 by linear shift
                current_mean = scaled.mean()
                shift = 3.5 - current_mean
                adjusted = np.clip(scaled + shift, 0, 5)  # Keep in [0, 5] after shift

                pref[feasible > 0] = adjusted
            else:
                # If all values are the same, assign mean = 3.5
                pref[feasible > 0] = 3.5

        # no constraint on itself
        np.fill_diagonal(constraint, True)

        # edges F1*S1 x F1*S1
        edges = feasible.reshape(-1, 1) & constraint & feasible.reshape(1, -1)

        # Create edge features
        edge_features = np.zeros((edges.shape[0], edges.shape[1], 3))  # 3D edge features for taxi, laps, reflective

        # Fill edge features based on the constraints
        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                if edges[i, j]:  # If there is an edge
                    edge_features[i, j, 0] = 1 if taxi[i, j] else 0  # Taxi feature
                    edge_features[i, j, 1] = 1 if laps[i, j] else 0  # Laps feature
                    edge_features[i, j, 2] = 1 if reflective[i, j] else 0  # Reflective feature

        G = (pref,feasible,edges, edge_features)

        masks = np.zeros(group.size, dtype=np.bool_)
        masks[sites] = True
        # F1 xS, F1*S1 x F1*S1, F1, S1
        return dispatch.feasible[0, crafts] & masks,G, crafts, sites


    @staticmethod
    def correlate(r, dispatch, crafts, sites, overlap=False):
        # activate the rule
        if r.data is None: r(dispatch.flags, dispatch.assignments, 0, 0)
        # craft ids B x F x F \ in {0,...C-1},  2 x h x S, C x h
        craft_ids, centers, corr = r.craft.data[-1], r.criteria.data[0], r.data
        # F1 x F1
        ids = np.reshape(craft_ids, (-1, craft_ids.shape[-1]))[crafts, :][:, crafts]
        # h x S1 x1, h x 1 x S1 -> h x S1 x S1
        centers = [centers[0][:, sites][..., np.newaxis] & centers[1][:, sites][:, np.newaxis, :],
                   centers[1][:, sites][..., np.newaxis] & centers[0][:, sites][:, np.newaxis, :]]
        # overlaps, overlaps at the site itself
        if overlap: centers = [np.stack([np.eye(sites.size, dtype=np.bool_)] * centers[0].shape[0], axis=0)] * 2
        # C x h, h x s1^2  -> C x S1^2 -> C x S1 x S1
        centers = [np.reshape(corr @ cc.reshape((cc.shape[0], -1)), (-1, *cc.shape[1:])) for cc in centers]
        #  F1*F1 X S1 X S1 -> F1 x F1 X S1 x S1
        hits = [cc[d.reshape(-1)].reshape((*d.shape, *cc.shape[1:])) for d, cc in zip((ids, ids.T), centers)]
        # reshape  F1 x F1 x S1 x S1 -> F1 x S1 x F1 x S1 -> F1*S1 x F1*S1
        return [h.transpose(0, 2, 1, 3).reshape(-1, crafts.size * sites.size) for h in hits]

    def pref(self, r, dispatch, crafts, sites):
        # compute the pref with the already assignments B x F
        masks = (dispatch.flags & dispatch.assignments > 0)
        # cold start
        if not np.any(masks): return [np.full((crafts.size, sites.size), 0.25)] * 2
        # activate the rule
        if r.data is None: r(dispatch.flags, dispatch.assignments, 0, None)
        # craft ids B x F x F \ in {0,...C-1},  2 x h x S, C x h
        craft_ids, centers, corr = r.craft.data[-1], r.criteria.data[0], r.data
        # k x F1, k x F1
        l2r, r2l = craft_ids[masks][:, crafts], (craft_ids.swapaxes(-2, -1)[masks][:, crafts]).swapaxes(-2, -1).T
        # k,
        assignments, idx = np.unique(dispatch.assignments[masks], return_inverse=True)
        # (h x Sd ->h) & (h x S1 -> h)
        ms = [np.any(centers[1][:, assignments], axis=-1) & np.any(centers[0][:, sites], axis=-1),
              np.any(centers[0][:, assignments], axis=-1) & np.any(centers[1][:, sites], axis=-1)]
        # further filtering,  k*F1 x h -> h
        filters = [np.any(corr[ids.reshape(-1)], axis=0) for ids in (l2r, r2l)]
        # masks
        ms = [m & _m for m, _m in zip(ms, filters)]
        # h1 x Sd  x S1,  h1 x S1 x Sd
        cc = [centers[0, ms[0]][:, assignments][..., np.newaxis] & centers[1, ms[0]][:, sites][:, np.newaxis, :],
              centers[0, ms[1]][:, sites][..., np.newaxis] & centers[1, ms[1]][:, assignments][:, np.newaxis, :]]
        # C x h1, h1 x Sd x S1 -> C x Sd x S1
        # C x h1, h1 x S1 x Sd -> C x S1 x Sd -> C x Sd x S1
        data = [np.reshape(d[:, m] @ c.reshape((c.shape[0], -1)), (-1, *c.shape[1:])).transpose(dim) \
                for d, m, c, dim in zip((corr, corr), ms, cc, ((0, 1, 2), (0, 2, 1))) if np.any(c)]
        # k x F1
        idx = np.tile(idx[:, np.newaxis], (1, crafts.size))
        # k x F1 X S1, k x F1 x S1
        pref = [d[l.reshape(-1), idx.reshape(-1)].reshape((*l.shape, -1)) for d, l in zip(data, (l2r, r2l))]
        # k x F1 x S1 -> F1 x S1
        return [np.sum(p, axis=0) for p in pref] + [np.full((crafts.size, sites.size), 1e-5)]

    def scatter(self, r, dispatch, crafts, sites):
        # compute the pref with the already assignments B x F
        masks = (dispatch.flags & dispatch.assignments > 0)
        # cold start
        if not np.any(masks): return [np.full((crafts.size * sites.size), 0.0)] * 2
        # activate the rule
        if r.data is None: r(dispatch.flags, dispatch.assignments, 0, 0)
        # craft ids B x F x F \ in {0,...C-1},  C x (C+1)
        craft_ids, corr = r.craft.data[-1], r.data
        # F1 x F1
        ids = np.reshape(craft_ids, (-1, craft_ids.shape[-1]))[crafts, :][:, crafts]
        # F1 x F1
        benefit = np.reshape(corr[ids.reshape(-1)], ids.shape)
        # F1 x F1 X 1 x1 , 1 x 1 x S1 x S1 -> F1 x F1 x S1 x S1
        values = benefit[..., np.newaxis, np.newaxis] * np.eye(sites.size, dtype=np.float_)[np.newaxis, np.newaxis, ...]
        # F1*S1, F1*S1
        return [h.transpose(0, 2, 1, 3).reshape(-1, crafts.size * sites.size) for h in (values, values)]

def restore_vertex(x, site_size, crafts, sites):
    a = x // site_size
    b = x % site_size
    return (crafts[a], sites[b])

def fast_clustering_coefficient(adj):
    assert (adj == adj.T).all(), "邻接矩阵必须是对称的！"
    degrees = adj.sum(axis=1)
    A2 = np.dot(adj, adj)
    A3 = np.dot(A2, adj)
    triangles = np.diag(A3) / 2
    possible_triplets = degrees * (degrees - 1) / 2

    with np.errstate(divide='ignore', invalid='ignore'):
        clustering = np.where(possible_triplets > 0, triangles / possible_triplets, 0.0)
    return clustering

def normalize_features(feature_tensor: torch.Tensor) -> torch.Tensor:
    """
    对 feature_tensor 按列归一化，使每列所有节点特征和为1。
    参数:
        feature_tensor: [num_entities, feat_dim] 张量
    返回:
        归一化后的张量，形状同输入
    """
    col_sum = feature_tensor.sum(dim=0, keepdim=True)  # shape (1, feat_dim)
    # 避免除以零，col_sum为0时用1代替，保持该列不变
    col_sum[col_sum == 0] = 1
    normalized = feature_tensor / col_sum
    return normalized

def average_neighbor_degree(adj):
    degrees = adj.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_neigh_deg = np.where(degrees > 0, (adj @ degrees) / degrees, 0)
    return avg_neigh_deg

def construct_graph_data(pref, mask, adj_matrix, edge_attr):
    num_nodes = pref.shape[0]

    # 节点基础特征软偏好 (num_nodes, num_feat)
    x = torch.tensor(pref, dtype=torch.float).view(num_nodes, -1)

    # 计算度
    #  度 (degree)
    degree = adj_matrix.sum(axis=1)
    degree_t = torch.tensor(degree, dtype=torch.float).view(num_nodes, 1)

    # 聚类系数和三角形数
    clustering = fast_clustering_coefficient(adj_matrix)
    clustering_t = torch.tensor(clustering, dtype=torch.float).view(num_nodes, 1)

    # 平均邻居度

    avg_neigh_deg = average_neighbor_degree(adj_matrix)
    avg_neigh_deg_t = torch.tensor(avg_neigh_deg, dtype=torch.float).view(num_nodes, 1)

    # 拼接所有节点特征
    x = torch.cat([x, degree_t, clustering_t, avg_neigh_deg_t], dim=1)

    # 归一化所有节点特征（按列）
    x = normalize_features(x)

    # 边索引
    edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)

    # 边特征
    num_edges = edge_index.shape[1]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    edge_attr_3 = torch.zeros((num_edges, edge_attr.shape[2]), dtype=torch.float)
    for i in range(num_edges):
        src, dst = int(edge_index[0][i]), int(edge_index[1][i])
        edge_attr_3[i] = edge_attr[src, dst]

    # weighted
    total_benefits = np.sum(pref)
    if total_benefits > 0:
        pref_normalized = pref / total_benefits
    else:
        raise ValueError("总和为零，无法进行归一化。")

    pref_normalized = torch.tensor(pref_normalized, dtype=torch.float).view(-1, 1)

    # 构造Q矩阵
    Q = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                Q[i][j] = 0
            elif adj_matrix[i][j] == 1:
                Q[i][j] = 1
    sum = Q.sum() + 1e-10
    if sum == 0:
        print(sum)
    Q_normalized =2 * Q / sum  # 除以和
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                Q_normalized[i][j] = -pref_normalized[i]

    # graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_3, pref=pref_normalized, Q=Q_normalized)
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_3, Q=Q_normalized)
    return graph_data


class GNPOptimizer:
    def __init__(self, Policy, n_positions, pretrained=None):
        self.Policy = Policy
        self.n_positions = n_positions
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if pretrained:
            print("加载模型")
            state_dict = torch.load(pretrained, weights_only=True)  # 显式启用 safer 模式
            self.Policy.model.load_state_dict(state_dict)


    def __call__(self, dispatch, feasible, G, crafts, sites):
        site_size = len(sites)
        pref, mask, adj_matrix, edge_attr = G

        num_nodes = len(pref)
        if num_nodes == 1:
            # 跳过节点数为1的图
            return [(crafts[0],sites[0])],[]
        # 构造图神经网络数据集
        num_nodes = pref.shape[0]
        # Convert pref to tensor
        pref = torch.tensor(pref, dtype=torch.float).view(num_nodes, -1)

        # 计算度
        #  度 (degree)
        degree = adj_matrix.sum(axis=1)
        degree_t = torch.tensor(degree, dtype=torch.float).view(num_nodes, 1)

        # 聚类系数和三角形数
        clustering = fast_clustering_coefficient(adj_matrix)
        clustering_t = torch.tensor(clustering, dtype=torch.float).view(num_nodes, 1)

        # 平均邻居度

        avg_neigh_deg = average_neighbor_degree(adj_matrix)
        avg_neigh_deg_t = torch.tensor(avg_neigh_deg, dtype=torch.float).view(num_nodes, 1)

        # 拼接所有节点特征
        x = torch.cat([pref, degree_t, clustering_t, avg_neigh_deg_t], dim=1)

        # 归一化所有节点特征（按列）
        x = normalize_features(x)

        # 边索引
        edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)

        # 边特征
        num_edges = edge_index.shape[1]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        edge_attr_3 = torch.zeros((num_edges, edge_attr.shape[2]), dtype=torch.float)
        for i in range(num_edges):
            src, dst = int(edge_index[0][i]), int(edge_index[1][i])
            edge_attr_3[i] = edge_attr[src, dst]

        # graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_3, pref=pref_normalized, Q=Q_normalized)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_3)

        data.to(self.device)

        probs = self.Policy.model(data.x, data.edge_index, data.edge_attr)

        mis = self.find_mis(adj_matrix, probs)

        # 确保返回的团与sites对应
        if mis:
            restored_clique = {restore_vertex(x, site_size, crafts, sites) for x in mis}
        else:
            restored_clique = set()  # 如果没有找到团，则返回空集

        return restored_clique,[]

    def find_mis(self, adj_matrix, node_probabilities, threshold=0.5):
        max_independent_set = []
        selected_nodes = set()

        # Ensure node_probabilities is 1-dimensional
        node_probabilities = node_probabilities.squeeze()  # Convert to 1-dimensional tensor

        # Sort nodes by probability in descending order
        sorted_indices = torch.argsort(node_probabilities, descending=True).tolist()
        return sorted_indices
        for node in sorted_indices:
            # Filter out nodes with probability less than the threshold
            if node_probabilities[node] < threshold:
                continue

            # Check if the current node is independent of all the nodes in the current independent set
            if self.is_independent(adj_matrix, max_independent_set, node):
                max_independent_set.append(node)
                selected_nodes.add(node)

        if len(max_independent_set) == 0:
            print("No independent set found that satisfies the conditions.")

        return max_independent_set

    def is_independent(self, adj_matrix, independent_set, node):
        for n in independent_set:
            if adj_matrix[node, n] == 1:
                return False
        return True



class MISGreedyOptimizer:
    def __init__(self, n_positions, pretrained=None):
        self.n_positions = n_positions

    def __call__(self, dispatch, feasible, graph, crafts, sites):
        site_size = len(sites)
        # 使用贪心算法求解最大独立集
        mis = self.solve_max_set(graph)

        if mis:
            restored_clique = {restore_vertex(x, site_size, crafts, sites) for x in mis}
            return restored_clique, (graph, {node: 1 for node in mis})  # 返回选中的节点标签为1
        else:
            # 全0标签表示没有节点被选中
            label = {node: 0 for node in range(len(graph[0]))}
            return set(), (graph, label)

    def solve_max_set(self, G):
        node_weight, feasible, edges, _ = G

        # 创建图
        G_nx = nx.Graph()

        # 添加节点和边
        for i in range(len(node_weight)):
            if feasible[i]:  # 只添加可行的节点
                G_nx.add_node(i, weight=node_weight[i])  # 将权重也添加掉，虽然 max_independent_set 不使用它

                # 添加边（原图边）
                for j in range(len(edges)):
                    if edges[i, j] == 1 and i != j:  # 如果在原图中有边，添加这条边
                        G_nx.add_edge(i, j)

        # 使用 networkx 的 maximal_independent_set 函数求解最大独立集
        independent_set = nx.maximal_independent_set(G_nx)

        return list(independent_set)  # 返回最大独立集的节点列表

class MISGurobiOptimizer:
    def __init__(self, n_positions, pretrained=None):
        self.n_positions = n_positions

    def __call__(self, dispatch, feasible, graph, crafts, sites, alpha=1, beta=1, gamma=0.0, time_limit=3600):
        site_size = len(sites)

        mis, label,val = self.solve_max_weighted_set(graph, alpha, beta, gamma, time_limit)

        if mis:
            restored_clique = {restore_vertex(x, site_size, crafts, sites) for x in mis}
            return restored_clique, (graph, label,val)
        else:
            # 全0标签表示没有节点被选中
            label = {node: 0 for node in range(len(graph['pref_normalized']))}
            return set(), (graph, label,0)

    def solve_max_weighted_set(self, G, alpha, beta, gamma, time_limit):
        node_weight, feasible, edges, edge_features = G

        # Create Gurobi model
        model = gp.Model("max_weighted_independent_set")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = time_limit

        # Create decision variables
        n_nodes = len(node_weight)
        x = model.addVars(n_nodes, vtype=gp.GRB.BINARY, name='x')

        # Set the objective function: maximize the weighted sum
        obj_linear = alpha * gp.quicksum(x[i] for i in range(n_nodes)) + beta * gp.quicksum(
            node_weight[i] * x[i] for i in range(n_nodes))
        model.setObjective(obj_linear, sense=gp.GRB.MAXIMIZE)

        # Add constraints: for each edge, the two adjacent nodes cannot be selected simultaneously
        for i in range(n_nodes):
            for j in range(n_nodes):
                if edges[i, j] and i != j:  # Ensure i and j are different
                    model.addConstr(x[i] + x[j] <= 1)

                    # Add feasibility constraints
        for i in range(n_nodes):
            if not feasible[i]:
                model.addConstr(x[i] == 0)  # If the node is not feasible, force its decision variable to 0

        # Solve the model
        model.optimize()

        # Get the results
        if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.TIME_LIMIT:
            selected_nodes = [i for i in range(n_nodes) if x[i].X > 0.5]
            obj_val = model.ObjVal
            label = [x[i].X if i in selected_nodes else 0 for i in range(n_nodes)]

            # Sort the selected nodes by their node weights in descending order
            # sorted_selected_nodes = sorted(selected_nodes, key=lambda x: node_weight[x], reverse=True)

            return selected_nodes, label,obj_val
        else:
            return [], {},0





