import torch
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from core_qnn.quaternion_ops import *
from sklearn.cluster import KMeans
from metrics import cal_clustering_metric


def get_Laplacian_from_weights(weights):
    degree = torch.sum(weights, dim=1).pow(-0.5)
    return (weights * degree).t() * degree

def to_tensor(X):
    if isinstance(X, torch.Tensor):
        return X
    return torch.Tensor(X)

def get_Laplacian(A):
    device = A.device
    dim = A.shape[0]
    L = A + torch.eye(dim).to(device)
    D = L.sum(dim=1)
    sqrt_D = D.pow(-1 / 2)
    Laplacian = sqrt_D * (sqrt_D * L).t()
    return Laplacian



class QGNNLayer(Module):
    def __init__(self, in_features, out_features, quaternion_ff=True,
                 act=F.relu, init_criterion='he', weight_init='quaternion',
                 seed=None):
        super(QGNNLayer, self).__init__()
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        self.quaternion_ff = quaternion_ff
        self.act = act
        if quaternion_ff:
            self.r = Parameter(torch.Tensor(self.in_features, self.out_features))
            self.i = Parameter(torch.Tensor(self.in_features, self.out_features))
            self.j = Parameter(torch.Tensor(self.in_features, self.out_features))
            self.k = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.commonLinear = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = np.random.RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        if self.quaternion_ff:
            winit = {'quaternion': quaternion_init,
                     'unitary': unitary_init}[self.weight_init]
            affect_init(self.r, self.i, self.j, self.k, winit,
                        self.rng, self.init_criterion)
        else:
            stdv = math.sqrt(6.0 / (self.commonLinear.size(0) + self.commonLinear.size(1)))
            self.commonLinear.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        if self.quaternion_ff:
            r1 = torch.cat([self.r, -self.i, -self.j, -self.k], dim=0)
            i1 = torch.cat([self.i,  self.r, -self.k,  self.j], dim=0)
            j1 = torch.cat([self.j,  self.k,  self.r, -self.i], dim=0)
            k1 = torch.cat([self.k, -self.j,  self.i,  self.r], dim=0)
            cat_kernels_4_quaternion = torch.cat([r1, i1, j1, k1], dim=1)
            mid = torch.mm(x, cat_kernels_4_quaternion)
        else:
            mid = torch.mm(x, self.commonLinear)
        out = torch.mm(adj, mid)
        return self.act(out)



class GOAH(Module):
    def __init__(self, name, X, A, labels,
                 decomposition='symeig', is_norm=True,
                 layers=None, acts=None,
                 max_epoch=20, max_iter=50,
                 learning_rate=1e-2, coeff_reg=1e-3,
                 seed=114514, lam=-1,
                 hyper_k=8, hyper_n_clusters=50, hyper_S=2,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(GOAH, self).__init__()
        self.name = name
        self.device = device
        self.X = to_tensor(X).to(self.device)
        self.adjacency = to_tensor(A).to(self.device)
        self.labels = to_tensor(labels).to(self.device)
        self.decomposition = decomposition
        self.is_norm = is_norm
        self.n_clusters = self.labels.unique().shape[0]
        self.layers = layers if layers is not None else [32, 16]
        self.acts = acts
        self.max_epoch = max_epoch
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.coeff_reg = coeff_reg
        self.seed = seed
        self.data_size = self.X.shape[0]
        self.input_dim = self.X.shape[1]
        self.lam = lam
        self.hyper_k = hyper_k
        self.hyper_n_clusters = hyper_n_clusters
        self.hyper_S = hyper_S
        self.lam_h = 0.03
        self.hyper_decay_epoch = int(0.5 * self.max_epoch)
        self.current_epoch = 0
        self._build_up()
        self.to(self.device)


    def _build_up(self):
        self.pro1 = torch.nn.Linear(self.input_dim, self.layers[0])
        self.pro2 = torch.nn.Linear(self.input_dim, self.layers[0])
        self.pro3 = torch.nn.Linear(self.input_dim, self.layers[0])
        self.pro4 = torch.nn.Linear(self.input_dim, self.layers[0])

        self.qgnn1 = QGNNLayer(self.layers[0] * 4, self.layers[1] * 4, quaternion_ff=True, \
                               act=self.acts[0], init_criterion='he', weight_init='quaternion', seed=self.seed)
        self.qgnn2 = QGNNLayer(self.layers[1] * 4, self.layers[2] * 4, quaternion_ff=True, \
                               act=self.acts[1], init_criterion='he', weight_init='quaternion', seed=self.seed)

    def forward(self, Laplacian):
        x1 = self.pro1(self.X)
        x2 = self.pro2(self.X)
        x3 = self.pro3(self.X)
        x4 = self.pro4(self.X)

        input = torch.cat((x1, x2, x3, x4), dim=1)

        input = self.qgnn1(input, Laplacian)
        input = self.qgnn2(input, Laplacian)
        input = input.reshape(self.data_size, 4, self.layers[2]).sum(dim=1) / 4.
        self.embedding = input

        recons_A = self.embedding.matmul(self.embedding.t())

        return recons_A

    def build_loss_reg(self):
        loss_reg = 0

        for module in self.modules():
            if type(module) is torch.nn.Linear:
                loss_reg += torch.abs(module.weight).sum()
            if type(module) is QGNNLayer:
                loss_reg += (
                            torch.abs(module.r) + torch.abs(module.i) + torch.abs(module.j) + torch.abs(module.k)).sum()

        return loss_reg



    def clustering(self, weights):
        degree = torch.sum(weights, dim=1).pow(-0.5)
        L = (weights * degree).t() * degree

        if self.decomposition == 'symeig':
            _, vectors = L.symeig(True)
        elif self.decomposition == 'svd':
            vectors, _, __ = torch.svd(L)
        elif self.decomposition == 'eigh':
            _, vectors = torch.linalg.eigh(L)

        indicator = vectors[:, -self.n_clusters:].detach()

        if torch.any(torch.isnan(indicator)):
            indicator = torch.nan_to_num(indicator, nan=0.0)

        if torch.any(torch.isinf(indicator)):
            indicator = torch.nan_to_num(indicator, posinf=1e6, neginf=-1e6)

        if self.is_norm:
            indicator = indicator / (indicator.norm(dim=1)+10**-10).repeat(self.n_clusters, 1).t()

        indicator = indicator.cpu().numpy()
        km = KMeans(n_clusters=self.n_clusters).fit(indicator)
        prediction = km.predict(indicator)
        acc, nmi, ari, f1 = cal_clustering_metric(self.labels.cpu().numpy(), prediction)

        return acc, nmi, ari, f1

    def run(self):
        acc_list = []
        nmi_list = []
        ari_list = []
        f1_list = []

        weights = self.update_graph(self.embedding)
        weights = get_Laplacian_from_weights(weights)

        acc, nmi, ari, f1 = self.clustering(weights)

        best_acc, best_nmi, best_ari, best_f1 = acc, nmi, ari, f1

        objs = []
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        Laplacian = get_Laplacian(self.adjacency)
        for epoch in range(self.max_epoch):
            self.current_epoch = epoch
            for i in range(self.max_iter):
                optimizer.zero_grad()
                recons_A = self(Laplacian)
                loss = self.build_loss(recons_A)
                loss.backward()
                optimizer.step()
                objs.append(loss.item())

            weights = self.update_graph(self.embedding)

            acc, nmi, ari, f1 = self.clustering(weights)
            loss = self.build_loss(recons_A)
            objs.append(loss.item())

            if acc >= best_acc:
                best_acc = acc
                best_nmi = nmi
                best_ari = ari
                best_f1 = f1
            acc_list.append(acc)
            nmi_list.append(nmi)
            ari_list.append(ari)
            f1_list.append(f1)

        acc_list = np.array(acc_list)
        nmi_list = np.array(nmi_list)
        ari_list = np.array(ari_list)
        f1_list = np.array(f1_list)
        return best_acc, best_nmi, best_ari, best_f1

    def build_pretrain_loss(self, recons_A):
        epsilon = torch.tensor(10 ** -7).to(self.device)
        recons_A = recons_A - recons_A.diag().diag()
        pos_weight = (self.data_size * self.data_size - self.adjacency.sum()) / self.adjacency.sum()
        loss = pos_weight * self.adjacency.mul((1 / torch.max(recons_A, epsilon)).log()) + (1 - self.adjacency).mul(
            (1 / torch.max((1 - recons_A), epsilon)).log())
        loss = loss.sum() / (loss.shape[0] * loss.shape[1])
        loss_reg = self.build_loss_reg()
        loss = loss + self.coeff_reg * loss_reg
        return loss

    def pretrain(self, pretrain_iter, learning_rate=None):
        learning_rate = self.learning_rate if learning_rate is None else learning_rate
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        Laplacian = get_Laplacian(self.adjacency)
        for i in range(pretrain_iter):
            optimizer.zero_grad()
            recons_A = self(Laplacian)
            loss = self.build_pretrain_loss(recons_A)
            loss.backward()
            optimizer.step()



    def update_graph(self, embedding):
        W_normal = embedding.matmul(embedding.t()).detach()

        from Quaternion_MLdata_load2 import build_dynamic_hypergraph

        emb_cpu = embedding.detach().cpu().numpy()

        dhg = build_dynamic_hypergraph(emb_cpu,
                                       k_neighbors=self.hyper_k,
                                       n_clusters=self.hyper_n_clusters,
                                       S=self.hyper_S,
                                       device=torch.device('cpu'))

        self.current_weights_hyper = dhg["W"].to(self.device)
        self.current_H = dhg["H"].to(self.device)
        self.current_hyperedge_sizes = dhg["hyperedge_sizes"].to(self.device)
        self.current_weights = W_normal
        return W_normal

    def update_graph2(self, embedding):
        W_normal = embedding.matmul(embedding.t()).detach()
        from Quaternion_MLdata_load3 import build_dynamic_hypergraph

        emb_cpu = embedding.detach().cpu().numpy()

        dhg = build_dynamic_hypergraph(
            emb_cpu,
            n_clusters=self.hyper_n_clusters,
            S=self.hyper_S,
            device=self.device,
            use_natural_neighbor=True,
        )

        self.current_weights_hyper = dhg["W"].to(self.device)
        self.current_H = dhg["H"].to(self.device)
        self.current_hyperedge_sizes = dhg["hyperedge_sizes"].to(self.device)
        self.current_weights = W_normal
        return W_normal

    def build_loss(self, recons_A):
        size = self.X.shape[0]
        eps = 1e-8

        pos_weight = (size * size - self.adjacency.sum()) / self.adjacency.sum()

        loss_recon = pos_weight * self.adjacency.mul(
            (1 / torch.clamp(recons_A, min=eps)).log()
        ) + (1 - self.adjacency).mul(
            (1 / torch.clamp(1 - recons_A, min=eps)).log()
        )
        loss_recon = loss_recon.sum() / (size ** 2)

        loss_reg = self.build_loss_reg()

        Z = self.embedding
        deg = torch.sum(self.adjacency, dim=1)
        L_n = torch.diag(deg) - self.adjacency
        loss_graph = torch.trace(Z.t() @ L_n @ Z) / size

        loss_hyper = 0.0
        if (
                hasattr(self, "current_weights_hyper")
                and self.current_weights_hyper is not None
                and self.current_epoch < self.hyper_decay_epoch
        ):
            lam_h = self.lam_h * (
                    1.0 - self.current_epoch / self.hyper_decay_epoch
            )

            W_h = self.current_weights_hyper
            L_h = hypergraph_laplacian(W_h)
            loss_hyper = lam_h * torch.trace(Z.t() @ L_h @ Z) / size

        loss = (
                loss_recon
                + self.coeff_reg * loss_reg
                + self.lam * loss_graph
                + loss_hyper
        )

        return loss



def hypergraph_laplacian(W):
    """
    Build (unnormalized) hypergraph Laplacian.
    W: (n, n)
    """
    deg = torch.sum(W, dim=1)
    L = torch.diag(deg) - W
    return L
