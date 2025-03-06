import tsplib95
import gurobipy as gp 
from gurobipy import GRB
from itertools import combinations
import networkx as nx
from networkx.utils import UnionFind
import numpy as np
import math
import copy
import random
import matplotlib
import matplotlib.pyplot as plt
import cProfile
matplotlib.use('Agg')

class Graph(object):
    def __init__(self, adj):
        self.adj = adj
    
    def number_of_nodes(self):
        return len(self.adj.keys())

    def get_edge_list(self):
        #output has two copies of each edge
        edges = []
        for k in self.adj.keys():
            for v in self.adj[k].keys():
                w = self.adj[k][v]
                edges.append([k, v, w])
        
        return edges 

    def merge(self, u, v):
        for k in self.adj[v].keys(): 
            if k != u:
                if k in self.adj[u]:
                    self.adj[u].update({k : self.adj[u][k] + self.adj[v][k]})
                else:
                    self.adj[u].update({k: self.adj[v][k]})
            
        del self.adj[u][v]
        del self.adj[v]

        for k in self.adj.keys():
            if v in self.adj[k].keys():
                w = self.adj[k][v]
                del self.adj[k][v]
                if u in self.adj[k].keys():
                    self.adj[k].update({u: self.adj[k][u] + w})
                else:
                    self.adj[k].update({u: w})
        
    def boost_edge(self, u, v, boost_factor):
        w = self.adj[u][v]
        w *= boost_factor
        self.adj[u].update({v: w})
        self.adj[v].update({u: w})

def convert_graph(graph):
    #converts nx.Graph to Graph
    adj = {}

    for v in graph.nodes():
        adj[v] = {}
    
    for u,v in graph.edges():
        w = graph[u][v]['weight']
        adj[u].update({v: w})
        adj[v].update({u: w})
    
    return Graph(adj)

def load_instance(filename):
    problem = tsplib95.load(filename)
    G = problem.get_graph()

    return G

def build_model(graph, RHS):
    w = {(u,v): graph[u][v]['weight'] for u,v in graph.edges()}
    m = gp.Model()
    vars = m.addVars(w.keys(), obj = w, vtype = GRB.CONTINUOUS, name = 'x', lb = 0, ub = 1)
    vars.update({(j,i): vars[i,j] for i,j in vars.keys()})
    cons = m.addConstrs(vars.sum(v, '*') == RHS for v in graph.nodes())

    return m, vars

def build_solution_graph(model, n):
    #returns a graph with optimal solution of model as vector of edge weights
    model.setParam(gp.GRB.Param.Method, 2)
    model.optimize()

    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    
    w = []
    for x in model.getVars():
        name, val = x.VarName, x.X
        l = len(name)
        name = name[2: l-1]
        u, v = name.split(',')
        w.append((int(u), int(v), val))

    G.add_weighted_edges_from(w)

    return G 

def add_subtour_constraint(model, variables, graph, partition):
    S, T = partition
    if len(S) >= 2:
        model.addConstr(gp.quicksum(variables[u,v] for u,v in combinations(S, 2) if (u,v) in graph.edges()) <= len(S)-1)
    if len(T) >= 2:
        model.addConstr(gp.quicksum(variables[u,v] for u,v in combinations(T, 2) if (u,v) in graph.edges()) <= len(T)-1)

    return model

def minimum_spanning_edges(G, weight = 'weight'):
    tree = []
    subtrees = UnionFind()
    edges = sorted(G.edges(data = True), key = lambda t: t[2].get(weight, 1))
    for u, v, d in edges:
        if subtrees[u] != subtrees[v]:
            tree.append((u, v))
            subtrees.union(u, v)
    
    return tree

def find_min_cut(graph):
    G = nx.Graph()
    edges = graph.get_edge_list()
    for u, v, w in edges:
        G.add_edge(u, v, weight = w)

    return nx.stoer_wagner(G)

def Karger(graph):
    G = nx.Graph()
    edges = graph.get_edge_list()
    for u, v, w in edges:
        if w != 0:
            g = np.random.gumbel(0, 1)
            G.add_edge(u, v, weight = -g - math.log(w))
    
    T = minimum_spanning_edges(G)
    n = len(G)
    F = nx.Graph(T[0:n-2])
    if len(F) != len(G):
        for v in G.nodes():
            F.add_node(v)
    
    S = [F.subgraph(c).copy() for c in nx.connected_components(F)]
    partition = (list(S[0]), list(S[1]))
    cut_size = 0
    for u, v, w in edges:
        if(u in partition[0] and v in partition[1]):
            cut_size += w
    
    return cut_size, partition

def boost_graph(graph, partition, boost_factor):
    adj = copy.deepcopy(graph.adj)
    G = Graph(adj)
    for i in range(len(partition)):
        S = partition[i]
        for u, v in list(combinations(S, 2)):
            if u in adj[v].keys():
                G.boost_edge(u, v, boost_factor)
    
    return G

def boost_edges(graph, edges, boost_factor):
    adj = copy.deepcopy(graph.adj)
    G = Graph(adj)

    for u,v in edges:
        if u in adj[v].keys():
            G.boost_edge(u, v, boost_factor)
    
    return G

def random_choice(arr):
    #picks a random element from arr with probability proportionate to the values of elements
    r = random.uniform(0, sum(arr))
    s = 0
    for i in range(len(arr)):
        s += arr[i]
        if s >= r:
            return i

def sample(edges, target):
    #randomly picks edges with probability proportionate to edge weights until the sum of weights passes target
    edges_copy = copy.deepcopy(edges)
    curr_sum = 0
    output = []

    weight_sum = sum([e[2] for e in edges])
    if target >= weight_sum:
        return edges

    while(curr_sum < target):
        weights = [e[2] for e in edges_copy]
        index = random_choice(weights)
        curr_sum += weights[index]
        output.append(edges_copy[index])
        del edges_copy[index]

    return output

def boost_graph_with_error(graph, partition, boost_factor, eta, rho):
    cut_edges, non_cut_edges = [], []
    S1, S2 = partition
    edges = graph.get_edge_list()
    for u,v,w in edges:
        if u in S1 and v in S2:
            cut_edges.append([u,v,w])
        if (u in S1 and v in S1) or (u in S2 and v in S2):
            if [v,u,w] not in non_cut_edges:
                non_cut_edges.append([u,v,w])

    cut_weight = sum([e[2] for e in cut_edges])
    non_cut_weight = sum([e[2] for e in non_cut_edges])

    fn, fp = eta*cut_weight, rho*cut_weight
    fp = min(fp, non_cut_weight)

    cut_sample = sample(cut_edges, fn)
    non_cut_sample = sample(non_cut_edges, fp)

    g = boost_graph(graph, partition, boost_factor)
    for u,v,w in cut_sample:
        g.adj[u].update({v: w*boost_factor})
        g.adj[v].update({u: w*boost_factor})
    for u,v,w in non_cut_sample:
        g.adj[u].update({v: w})
        g.adj[v].update({u: w})
    
    real_eta = sum([e[2] for e in cut_sample])/cut_weight
    real_rho = sum([e[2] for e in non_cut_sample])/cut_weight

    return g, real_eta, real_rho

def boosted_Karger(graph, boosted_graph):
    cut, partition = Karger(boosted_graph)
    S1, S2 = partition
    edges = graph.get_edge_list()
    cut_size = 0
    for u, v, w in edges:
        if u in S1 and v in S2:
            cut_size += w
    
    return cut_size, partition

def Karger_until_min_cut(graph, cut_size, reps, cap):
    run_time = []
    for i in range(reps):
        cnt = 1
        curr_cut = Karger(graph)[0]
        while(curr_cut > cut_size and cnt < cap):
            curr_cut = Karger(graph)[0]
            cnt += 1
        
        run_time.append(cnt)
        print(f"K {cnt}")
    
    return run_time

def boosted_Karger_until_min_cut(graph, boosted_graph, cut_size, reps, cap):
    run_time = []
    for i in range(reps):
        cnt = 1
        curr_cut = boosted_Karger(graph, boosted_graph)[0]
        while(curr_cut > cut_size and cnt < cap):
            curr_cut = boosted_Karger(graph, boosted_graph)[0]
            cnt += 1

        run_time.append(cnt)
        print(f"B {cnt}")
    
    return run_time

def add_edge_weight(G, u, v, w):
    w_prev = 0
    if G.has_edge(u, v):
        w_prev = G[u][v]["weight"]
    
    G.add_edge(u, v, weight = w_prev + w)
    
    return G

def add_cycle(G, perm):
    #adds the cycle given by perm to G
    for i in range(len(perm)):
        G = add_edge_weight(G, perm[i], perm[(i+1)%len(perm)], 1)
    
    return G 

def add_random_cycle(G, S):
    # adds a random Hamiltonian cycle in G over vertex set S
    perm = np.random.permutation(S)
    G = add_cycle(G, perm)
    
    return G

def build_synthetic_graph(n, k, r, eps):
    # returns a graph with n nodes, (S,T) is a partition of size (r, n-r)
    # the edges are the union of k Hamiltonian cycles that each cross (S,T) exactly twice
    # and k Hamiltonian cycles over each of S and T and eps*k random cycles over S and T

    G = nx.Graph()
    nodes = np.array(range(n))
    G.add_nodes_from(nodes)

    S, T = nodes[0: r], nodes[r: n]
    for i in range(k):
        G = add_random_cycle(G, S)
        G = add_random_cycle(G, T)

        PS, PT = np.random.permutation(S), np.random.permutation(T)
        perm = np.concatenate((PS, PT))
        G = add_cycle(G, perm)
    
    for i in range(int(eps*k)):
        partition = [S, T]
        side = partition[np.random.choice([0,1])]
        length = np.random.randint(3, len(side))
        cycle = np.random.choice(side, length)
        G = add_random_cycle(G, cycle)

    return G 

def build_matching_graph(n, k, t):
    #output has two partite sets of size n/2
    #edges are a union of k prefect matchings
    #one node's degree is smaller that others by t
    adj = {}
    for i in range(n):
        adj[i] = {}

    a = range(int(n/2))
    b = range(int(n/2), n)
    
    for i in range(k):
        l = np.random.permutation(a)
        r = np.random.permutation(b)
        for j in range(int(n/2)):
            u, v = l[j], r[j]
            if v in adj[u].keys():
                w = adj[u][v]
            else:
                w = 0
            adj[u].update({v: w+1})
            adj[v].update({u: w+1})
    
    sample = random.sample(sorted(adj[0].keys()), t)
    for s in sample:
        w = adj[0][s]
        if w == 1:
            del adj[0][s]
            del adj[s][0]
        else:
            adj[0].update({s: w-1})
            adj[s].update({0: w-1})
        
    return Graph(adj)

def build_graph(file):
    G = nx.Graph()
    with open(file) as f:
        lines = f.read().splitlines(True)
        for line in lines:
            l = line.strip().split(' ')
            edge = [int(x) for x in l]
            if len(edge) == 4:
                u, v, w, t = edge
                G = add_edge_weight(G, u, v, w)
            else:
                G = add_edge_weight(G, edge[0], edge[1], 1)
    return G 

def vertex_overlap(p1, p2):
    #p1 and p2 are two partitions
    S1, T1 = p1
    S2, T2 = p2

    cap1 = (list(set(S1) & set(S2)), list(set(T1) & set(T2)))
    cap2 = (list(set(S1) & set(T2)), list(set(S2) & set(T1)))

    size1 = len(cap1[0]) + len(cap1[1])
    size2 = len(cap2[0]) + len(cap2[1])

    if size1 >= size2:
        return (len(cap1[0]), len(cap1[1]))
    
    return (len(cap2[0]),len(cap2[1]))

def get_cut_edges(graph, partition):
    S, T = partition
    adj = graph.adj

    cut_edges = []
    for u in S:
        for v in T:
           if v in adj[u].keys():
               cut_edges.append(tuple(sorted((u, v))))

    return cut_edges 

def get_weighted_cut_edges(graph, partition):
    cut_edges = get_cut_edges(graph, partition)
    weighted_cut_edges = []
    adj = graph.adj

    for u,v in cut_edges:
        w = adj[u][v]
        weighted_cut_edges.append((u,v,w))
    
    return weighted_cut_edges

def edge_overlap(graph1, graph2, partition1, partition2):
    cut_edges_1 = get_cut_edges(graph1, partition1)
    cut_edges_2 = get_cut_edges(graph2, partition2)

    overlap = list(set(cut_edges_1) & set(cut_edges_2))

    return len(overlap)

def get_support(G):
    supp = []
    for u,v in G.edges():
        w = G[u][v]['weight']
        if w != 0:
            supp.append((u,v,w))
    
    return supp

def get_support_size(G):
    return len(get_support(G))

def get_frac_size(G):
    frac = [w for (u,v,w) in get_support(G) if w != 1]

    return len(frac)

def remove_zero_edges(G):
    for u,v in G.edges():
        if G[u][v]['weight'] == 0:
            G.remove_edge(u,v)
    
    return G

def load_data(filename):
    lines = []
    with open(filename, 'r') as file:
        for line in file:
            lines.append(line.strip())
    
    line = lines[1]
    line = line[9:-1]
    karger = [int(x) for x in line.split(',')]
    boosted = []
    for i in range(1, 22):
        line = lines[2*i+1]
        line = line[10:-1]
        boosted.append([int(x) for x in line.split(',')])
    
    line = lines[-1]
    line = line[11:-1]
    eta = [float(x) for x in line.split(',')]

    return karger, boosted, eta

def resample(data):
    sample = random.choices(data, k = len(data))
    
    return sum(sample)/len(data)

def bootstrap(data, N):
    means = []
    for i in range(N):
        means.append(resample(data))

    return means

def plot2(x, y1, y2, lower1, upper1, lower2, upper2, xlabel, ylabel, ylabel1, ylabel2, filename, title):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y1, color = 'indianred', label = ylabel1)
    plt.plot(x, y2, color = 'blue', label = ylabel2)
    plt.fill_between(x, lower1, upper1, color = 'indianred', alpha = 0.2)
    plt.fill_between(x, lower2, upper2, color = 'blue', alpha = 0.2)
    plt.title(title)
    plt.legend()
    plt.savefig(filename) 

def plot2_step_line(x, y1, y2, xlabel, ylabel, ylabel1, ylabel2, filename):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.step(x, y1, where = 'mid', color = 'indianred', label = ylabel1)
    plt.step(x, y2, where = 'mid', color = 'blue', label = ylabel2)
    plt.legend()
    plt.savefig(filename)

def run_with_subtour_constraint(n, k, r, eps):
    K, B = [], []
    with open("TSP LOG.txt", "w") as file:
        G = build_synthetic_graph(n, k, r, eps)
        m, vars = build_model(G, 2)
        m.Params.LogToConsole = 0

        file.write(f"n: {n}\n")
        file.write(f"m: {len(G.edges())}\n")
        file.write(f"k: {k}\n")
        file.write(f"eps: {eps}\n")
        file.write('-' * 30 + "\n")

        instances = []
        cnt = 20

        for j in range(cnt):
            H = build_solution_graph(m, n)
            min_cut, partition = nx.stoer_wagner(H)
            H = remove_zero_edges(H)
            m = add_subtour_constraint(m, vars, H, partition) 
            instances.append((copy.deepcopy(H), min_cut, copy.deepcopy(partition)))

            file.write(f"{j} ")

            if min_cut >= 2:
                break
        
        file.write("\n")
        for i in range(j+1):
            G1, p1 = instances[i][0], instances[i][2]
            min_cut = instances[i][1] 
            GG1 = convert_graph(G1)
            cut_edges = get_weighted_cut_edges(GG1, p1)

            file.write(f"G{i}\n")
            file.write(f"SUPPORT SIZE: {get_support_size(G1)}\n")
            file.write(f"FRACTIONAL EDGES: {get_frac_size(G1)}\n")
            file.write(f"MIN CUT PARTITION: ({len(p1[0])}, {len(p1[1])})\n")
            file.write(f"MIN CUT SIZE: {min_cut}\n")
            file.write(f"CUT EDGES: {cut_edges}\n")
            file.write(f"CONNECTED: {nx.is_connected(G1)}\n") 
            
            if min_cut != 0:
                supp_edges = get_support(G1)
                int_edges = [(e[0], e[1]) for e in supp_edges if e[2] == 1]
                reps, cap, boost_factor = 10, n**2, int(math.log(n))
                GB = boost_edges(GG1, int_edges, boost_factor)

                Karger_runtime = Karger_until_min_cut(GG1, min_cut, reps, cap)
                boosted_runtime = boosted_Karger_until_min_cut(GG1, GB, min_cut, reps, cap)

                K.extend(Karger_runtime)
                B.extend(boosted_runtime)

                file.write(f"KARGER: {Karger_runtime}\n")
                file.write(f"BOOSTED: {boosted_runtime}\n")
            
            file.write('*' *30 + "\n")
    
    with open("TSP REPS.txt", "w") as f:
        f.write(f"K: {sorted(K)}\n")
        f.write(f"B: {sorted(B)}\n")

def tsp_plot(threshold):
    lines = []
    with open("TSP REPS.txt", "r") as file:
        for line in file:
            lines.append(line.strip())
    
    K = [int(x) for x in lines[0][4:-1].split(',')]
    B = [int(x) for x in lines[1][4:-1].split(',')]
    
    index = 0
    for i in range(len(K)):
        if K[i] > threshold or B[i] > threshold:
            index = i
            break

    K = K[index:]
    B = B[index:]

    values = copy.deepcopy(K)
    values.extend(B)
    values = sorted(list(set(values))) 

    K_cnt, B_cnt = [], []
    for val in values:
        temp = 0
        for k in K:
            if k <= val:
                temp += 1
        
        K_cnt.append(temp)
        temp = 0
        for b in B:
            if b <= val:
                temp += 1
        
        B_cnt.append(temp)

    plot2_step_line(values, K_cnt, B_cnt, "Number of Repetitions", "Number of Instances Solved", "Karger", "Boosted", f"TSP REPS - {threshold}.pdf")

def run_matching_graph(n, k, t, rho):
    with open(f"Matching Graph - eta (rho = {rho}).txt", "w") as file:
        G = build_matching_graph(n, k, t)
        min_cut, partition = find_min_cut(G)
        file.write(f"MIN CUT: {min_cut}\n")

        reps, cap = 30, n**2
        Karger_run_time = Karger_until_min_cut(G, min_cut, reps, cap)
        file.write(f"KARGER: {Karger_run_time}\n")

        eta = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]  
        real_eta = [] 
        for a in eta:
            file.write(f"eta: {a} ")
            boost_factor = n
            GB, a1, a2 = boost_graph_with_error(G, partition, boost_factor, a, rho)
            real_eta.append(a1)
            file.write(f"real: {a1}\n")
            boosted_run_time = boosted_Karger_until_min_cut(G, GB, min_cut, reps, cap)
            file.write(f"BOOSTED: {boosted_run_time}\n")
        
        file.write(f"real eta: {real_eta}")

def matching_graph_plot(rho):
    filename = f"Matching Graph - eta (rho = {rho})."
    karger, boosted, eta = load_data(filename + "txt")
    N = 100000
    x_len = len(boosted)

    karger_means = bootstrap(karger, N)
    boosted_means = [bootstrap(boosted[i], N) for i in range(len(boosted))]    

    karger_mean = [sum(karger_means)/len(karger_means)]*x_len
    boosted_mean = [sum(boosted_means[i])/len(boosted_means[i]) for i in range(len(boosted_means))]

    karger_percentile = np.percentile(karger_means, [2.5,97.5])
    boosted_percentile = [np.percentile(boosted_means[i], [2.5,97.5]) for i in range(len(boosted_means))]
    karger_lower = [karger_percentile[0]]*x_len
    karger_upper = [karger_percentile[1]]*x_len
    boosted_lower = [boosted_percentile[i][0] for i in range(len(boosted_means))]
    boosted_upper = [boosted_percentile[i][1] for i in range(len(boosted_means))]

    plot2(eta, karger_mean, boosted_mean, karger_lower, karger_upper, boosted_lower, boosted_upper, '$\\eta$', "Number of Repetitions", "Karger", "Boosted", filename + "pdf", f'$\\rho$ = {rho}') 

def run_real_datasets(filename, eps, k):
    with open(filename + " (Results).txt", "w") as file:
        G = build_graph(filename)
        n, m = G.number_of_nodes(), G.number_of_edges()
        edges = []
        for u,v in G.edges():
            edges.append((u,v))

        sample_size = int(eps*m)
        sample_indices = random.sample(range(m), sample_size)
        sample_edges = [edges[i] for i in sample_indices]
        sample_graph = nx.Graph()
        for u,v in sample_edges:
            w = G[u][v]['weight']
            sample_graph.add_edge(u, v, weight = w)

        SG = convert_graph(sample_graph)
        edge_set = set()
        for i in range(k):
            k_min, k_partition = Karger(SG)
            cut_edges = set(get_cut_edges(SG, k_partition))
            edge_set.update(cut_edges)

        edges_to_boost = []
        for u,v in edges:
            if (min(u,v), max(u,v)) not in edge_set:
                edges_to_boost.append((min(u,v), max(u,v)))

        GG = convert_graph(G)
        reps, cap, boost_factor = 100, 10*n, n
        GB = boost_edges(GG, edges_to_boost, boost_factor)
        
        min_cut, partition = nx.stoer_wagner(G)
        B = boosted_Karger_until_min_cut(GG, GB, min_cut, reps, cap)
        K = Karger_until_min_cut(GG, min_cut, reps, cap)
        file.write(f"K: {K}\n")
        file.write(f"B: {B}\n")

def real_data_plots():
    files = ["sanr400-0-7.mtx (Results).txt", "bn-mouse_brain_1.edges (Results).txt", "frb30-15-5.mtx (Results).txt"]
    K , B = [], []
    for file in files:
        with open(file, 'r') as file:
            karger = [float(num) for num in file.readline().strip()[4:-1].split(',')]
            boosted = [int(num) for num in file.readline().strip()[4:-1].split(',')]

            K.append(karger)
            B.append(boosted)

    N = 100000
    K_bootstrap, B_bootstrap = [], []
    for i in range(len(files)):
        K_bootstrap.append(bootstrap(K[i], N))
        B_bootstrap.append(bootstrap(B[i], N))

    datasets = ['sanr400-0-7', 'bn-mouse_brain_1', 'frb30-15-5']
    data_A = K_bootstrap
    data_B = B_bootstrap

    percentiles_A = [(np.percentile(data, 2.5), np.percentile(data, 97.5)) for data in data_A]
    percentiles_B = [(np.percentile(data, 2.5), np.percentile(data, 97.5)) for data in data_B]

    bar_width = 0.05
    bar_spacing = 0.1  

    fig, axes = plt.subplots(1, 3, figsize = (10, 6))
    T_width = 0.04

    for i in range(len(datasets)):
        ax = axes[i]

        ax.bar(i - bar_width/2 - bar_spacing/2, percentiles_A[i][1] - percentiles_A[i][0], 
            bottom=percentiles_A[i][0], width=bar_width, label=f'Karger', color='indianred')
        ax.bar(i + bar_width/2 + bar_spacing/2, percentiles_B[i][1] - percentiles_B[i][0], 
            bottom=percentiles_B[i][0], width=bar_width, label=f'Boosted', color='blue')

        ax.plot([i - bar_width/2 - bar_spacing/2, i - bar_width/2 - bar_spacing/2], 
                [min(data_A[i]), max(data_A[i])], color='indianred', lw=2) 
        ax.plot([i - bar_width/2 - bar_spacing/2 - T_width, i - bar_width/2 - bar_spacing/2 + T_width], 
                [max(data_A[i]), max(data_A[i])], color='indianred', lw=2) 
        ax.plot([i - bar_width/2 - bar_spacing/2 - T_width, i - bar_width/2 - bar_spacing/2 + T_width], 
                [min(data_A[i]), min(data_A[i])], color='indianred', lw=2) 
        
        ax.plot([i + bar_width/2 + bar_spacing/2, i + bar_width/2 + bar_spacing/2], 
                [min(data_B[i]), max(data_B[i])], color='blue', lw=2)  
        ax.plot([i + bar_width/2 + bar_spacing/2 - T_width, i + bar_width/2 + bar_spacing/2 + T_width], 
                [max(data_B[i]), max(data_B[i])], color='blue', lw=2)  
        ax.plot([i + bar_width/2 + bar_spacing/2 - T_width, i + bar_width/2 + bar_spacing/2 + T_width], 
                [min(data_B[i]), min(data_B[i])], color='blue', lw=2) 

        ax.set_title(f'{datasets[i]}')
        ax.set_ylabel('Number of Repetitions')
        ax.set_xticks([i - bar_width/2 - bar_spacing/2, i + bar_width/2 + bar_spacing/2])
        ax.set_xticklabels(['Karger', 'Boosted'])
        caption = ["k = 250", "k = 55", "k = 300"]
        ax.text(0.5, -0.1, caption[i], ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    plt.savefig("Read Datasets (2.5, 97.5).pdf")

def main():
    # First set of experiments:
    rho = [0, 10, 100]
    for r in rho:
        run_matching_graph(600, 100, 10, r)
        matching_graph_plot(r)

    # Second set of experiments:
    run_with_subtour_constraint(500, 50, 250, 1/2)
    tsp_plot(0)
    
    # Third set of experiments:
    run_real_datasets("frb30-15-5.mtx", 0.5, 300)
    run_real_datasets("sanr400-0-7.mtx", 0.5, 250)
    run_real_datasets("bn-mouse_brain_1.edges", 0.5, 55)
    real_data_plots()

if __name__ == "__main__":
    main()