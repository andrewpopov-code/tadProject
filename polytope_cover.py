import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
import os


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, labels: list[str], transforms: list = (), subset: str = None):
        super().__init__("./", download=True)
        self.transforms = transforms
        self._count = None
        self._labels = labels

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                l = [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj if line.strip().split('/')[0] in labels]
                return l

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes and w.split('/')[2] in labels]

        self._ix = list(range(len(self._walker)))

    @property
    def count(self):
        if self._count is not None:
            return self._count

        self._count = [0] * len(self._labels)
        with self:
            for _, l in self:
                self._count[l] += 1
        return self._count

    def transform(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __getitem__(self, n: int):
        return super().__getitem__(self._ix[n])

    def remove(self, ix: list):
        for i in ix:
            self._ix.remove(i)

    def __iter__(self):
        return (
            (self.transform(x[0]), self._labels.index(x[2])) for x in (self[i] for i in range(len(self)))
        )

    def __len__(self):
        return len(self._ix)


class PolytopeModel(nn.Module):
    def __init__(self, d, m, nP, nQ, l, l0, l1, c0, c1):
        """
        The model from paper on polytope basis cover
        :param d: data dimension
        :param m: number of hyperplanes
        :param nP: number of +1 polyhedrons
        :param nQ: number of -1 polyhedrons
        :param l: lambda
        """
        super().__init__()
        # self.linear1 = nn.Linear(d, m, bias=True)
        # self.linear2 = nn.Linear(m, nP + nQ, bias=False)

        self.hyperplanes = nn.Linear(d, m, bias=True)
        self.polytopes = nn.ModuleList([
            nn.Linear(m, 1, bias=False) for _ in range(nP + nQ)
        ])
        self.l0, self.l1 = nn.Parameter(torch.tensor(l0), requires_grad=False), nn.Parameter(torch.tensor(l1), requires_grad=False)
        self.c0, self.c1 = nn.Parameter(torch.tensor(c0), requires_grad=False), nn.Parameter(torch.tensor(c1), requires_grad=False)

        with torch.no_grad():
            # self.linear2.weight = -torch.sqrt(
            #     torch.square(torch.norm(self.linear1.weight, dim=1)) + torch.square(self.linear1.bias)
            # )

            for T in self.polytopes:
                T.weight = -torch.sqrt(
                    torch.square(torch.norm(self.linear1.weight, dim=1)) + torch.square(self.linear1.bias)
                )

        self.l = nn.Parameter(torch.tensor(l), requires_grad=False)
        self.aj = nn.Parameter(torch.hstack([torch.ones(nP), -torch.ones(nQ)]).unsqueeze(0), requires_grad=False)

    def forward(self, x):
        # x.flatten(start_dim=1)
        polytopes = F.relu(self.t(x)) * self.aj
        return torch.sigmoid(-self.l / 2 + polytopes.sum(dim=-1))

    def t(self, x):
        x = F.relu(self.hyperplanes(x))
        return self.l + torch.hstack([T(x) for T in self.polytopes])

    def compress(self, T: nn.Linear, scale, data: SubsetSC):
        torch.set_grad_enabled(False)

        m = self.hyperplanes.weight.shape[1]
        K = []
        for k in range(m):
            for l in range(m):
                if k == l: continue

                wk, wl = self.hyperplanes.weight[k], self.hyperplanes.weight[l]
                bk, bl = self.hyperplanes.bias[k], self.hyperplanes.bias[l]

                ok = True
                for x, _ in data:
                    if (x * wl).sum() + bl > 0 >= (x * wk).sum() + bk:
                        ok = False
                        break
                if ok:
                    K.append(k)

        if K:
            k = torch.argmin((torch.abs(T.weight) * torch.norm(self.hyperplanes.weight, dim=-1))[K]).item()
            ix = torch.hstack([torch.arange(k), torch.arange(k + 1, m)])
            self.hyperplanes.weight = self.hyperplanes.weight[:, ix]
            self.hyperplanes.bias = self.hyperplanes.bias[ix]
            T.weight = T.weight[ix]
            m -= 1

        for k in range(m):
            wk, bk = self.hyperplanes.weight[k], self.hyperplanes.bias[k]
            for x, _ in data:
                if (x * wk).sum() + bk > 0 and 0 < T(x) < 1:
                    self.hyperplanes.weight[k] *= scale
                    self.hyperplanes.bias[k] *= scale
                    T.weight[k] *= scale
                    break

        torch.set_grad_enabled(True)
        return T

    def multi_compress(self, T: nn.Linear, scale, data: SubsetSC):
        ok = False
        while not ok:
            ok = True
            for S in self.polytopes:
                for x, _ in data:
                    y = F.relu(S(x))
                    if y != 0 and y != 1:
                        self.compress(T, scale, data)
                        ok = False

    def extract(self, scale, data_loader: DataLoader, data: SubsetSC, epochs: int):
        optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10

        for _ in range(epochs):
            self.train_step(data_loader, optimizer)
            for T in self.polytopes:
                self.compress(T, scale, data)
            scheduler.step()

        found = False
        for T in self.polytopes:
            for x, _ in data:
                if 0 < T(x) < 1:
                    found = True
                    self.multi_compress(T, scale, data)
                    break
            if found: break

    def train_step(self, data: DataLoader, optimizer: optim.Optimizer):
        for x, target in data:
            out = self(x)
            loss = self.loss(out.squeeze(), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def loss(self, out, target):
        def l(x, y):
            return torch.sum(x * y) + torch.sum((1 - x) * (1 - y))

        f = self.l0 * l(F.sigmoid(out[target == 0]), 0.0).sum() / self.c0 + self.l1 * l(F.sigmoid(out[target == 1]), 1.0).sum() / self.c1
        return -f


class PolytopeModel2(nn.Module):
    def __init__(self, bias, l0, l1, c0, c1, epochs):
        super().__init__()

        self.epochs = epochs
        self.l0, self.l1 = nn.Parameter(torch.tensor(l0), requires_grad=False), nn.Parameter(torch.tensor(l1),
                                                                                             requires_grad=False)
        self.c0, self.c1 = nn.Parameter(torch.tensor(c0), requires_grad=False), nn.Parameter(torch.tensor(c1),
                                                                                             requires_grad=False)
        self.bias = nn.Parameter(torch.tensor(bias), requires_grad=False)

    def train_model(self, h: nn.Linear, T: nn.Linear, data: DataLoader):
        model = nn.Sequential(h, T, nn.Sigmoid())
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10

        for _ in range(self.epochs):
            for x, target in data:
                out = model(x)
                loss = self.loss(out.squeeze(), target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
        it = model.modules()
        next(it)
        return next(it), next(it)

    def get_polytope(self, sign: int, abs_bias, d: int, width: int, scale, data: DataLoader, Cc: list):
        m = width
        bias = nn.Parameter(torch.tensor(sign * abs_bias), requires_grad=False)
        ok = False
        data.dataset._ix = Cc

        while not ok:
            # Initialize T
            h = nn.Linear(d, m, bias=True)
            T = nn.Linear(m, 1, bias=False)
            T.bias = bias
            with torch.no_grad():
                T.weight[0] = sign * torch.sqrt(
                    torch.square(torch.norm(h.weight, dim=1)) + torch.square(h.bias)
                )
            h, T = self.train_model(h, T, data)
            h, T = self.al2(h, T, scale, data, data.dataset, self.epochs)
            model = nn.Sequential(h, T)
            label = (sign + 1) // 2
            ok = all(x[1] == label or model(x[0]).squeeze() == bias for x in data.dataset)
            m += 1

        data.dataset._ix = list(range(self.c0.item() + self.c1.item()))
        A = [i for (i, x) in enumerate(data.dataset) if model(x[0]).squeeze() == bias]
        return A
    
    @staticmethod
    def al1(h: nn.Linear, T: nn.Linear, scale, data: SubsetSC):
        torch.set_grad_enabled(False)

        m = h.weight.shape[0]
        K = []
        for k in range(m):
            for l in range(m):
                if k == l: continue

                wk, wl = h.weight[k], h.weight[l]
                bk, bl = h.bias[k], h.bias[l]

                ok = True
                for x, _ in data:
                    if (x * wl).sum() + bl > 0 >= (x * wk).sum() + bk:
                        ok = False
                        break
                if ok:
                    K.append(k)

        if K:
            k = torch.argmin((torch.abs(T.weight[0]) * torch.norm(h.weight, dim=1))[K]).item()
            ix = torch.hstack([torch.arange(k), torch.arange(k + 1, m)])
            h.weight = h.weight[ix]
            h.bias = h.bias[ix]
            T.weight = T.weight[:, ix]
            m -= 1

        for k in range(m):
            wk, bk = h.weight[k], h.bias[k]
            for x, _ in data:
                if (x * wk).sum() + bk > 0 and 0 < T(h(x)).squeeze() < 1:
                    h.weight[k] *= scale
                    h.bias[k] *= scale
                    T.weight[:, k] *= scale
                    break

        torch.set_grad_enabled(True)
        return h, T

    @staticmethod
    def multi_compress(h: nn.Linear, T: nn.Linear, scale, data: SubsetSC):
        ok = False
        while not ok:
            ok = True
            for x, _ in data:
                y = F.relu(T(h(x))).squeeze()
                if y != 0 and y != 1:
                    PolytopeModel2.al1(h, T, scale, data)
                    ok = False

    @staticmethod
    def al2(h: nn.Linear, T: nn.Linear, scale, data_loader: DataLoader, data: SubsetSC, epochs: int):
        model = nn.Sequential(h, T, nn.Sigmoid())
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10

        for _ in range(epochs):
            PolytopeModel2.train_step(model, data_loader, optimizer)
            h, T = PolytopeModel2.al1(h, T, scale, data)
            scheduler.step()

        for x, _ in data:
            if 0 < T(h(x)).squeeze() < 1:
                PolytopeModel2.multi_compress(h, T, scale, data)
                break
        return h, T

    def al4(self, acc, d, width, scale, data: DataLoader):
        CP, CQ = [], []
        CPc, CQc = list(range(len(data.dataset))), list(range(len(data.dataset)))
        target = [x[1] for x in data.dataset]
        while self.accuracy(target, CP, CQ) < acc:
            CP.append(self.get_polytope(1, self.bias, d, width, scale, data, CPc))
            for x in CP[-1]:
                CPc.remove(x)
            CQ.append(self.get_polytope(-1, self.bias, d, width, scale, data, CQc))
            for x in CQ[-1]:
                CQc.remove(x)

        return CP, CQ

    @staticmethod
    def accuracy(target: torch.Tensor, CP: list, CQ: list):  # list of integers
        res = torch.zeros_like(target)
        for c in CP:
            res[c] += 1.0
        for c in CQ:
            res[c] -= 1.0
        out = F.sigmoid(res).round()
        return 1 - (out - target).sum() / target.shape[0]

    def loss(self, out, target):
        def l(x, y):
            return torch.sum(x * y) + torch.sum((1 - x) * (1 - y))

        f = self.l0 * l(F.sigmoid(out[target == 0]), 0.0).sum() / self.c0 + self.l1 * l(F.sigmoid(out[target == 1]), 1.0).sum() / self.c1
        return -f
