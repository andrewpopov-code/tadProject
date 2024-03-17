import torch

import bisect
from dataclasses import dataclass


@dataclass
class Simplex:
    idx: int
    diameter: float


class BinomialCoeffTable:
    def __init__(self, n, k):
        self.table = [0] * (n + 1) * (k + 1)

        self.offset = k + 1
        for i in range(n + 1):
            self.table[i * self.offset] = 1
            for j in range(1, min(i, k + 1)):
                self.table[i * self.offset + j] = self.table[(i - 1) * self.offset + j - 1] + self.table[(i - 1) * self.offset + j]
            if i <= k:
                self.table[i * self.offset + i] = 1

    def __call__(self, n, k):
        if n < k:
            return 0
        return self.table[n * self.offset + k]


class VietorisRipsComplex:
    def __init__(self, x: torch.Tensor, dim):
        super().__init__()
        self.dim = dim
        self.x = x
        self.n = x.shape[0]
        self.binom = BinomialCoeffTable(x.shape[0], 1)  # TODO: ???

    def get_max_vertex(self, idx, n, k):
        top, bottom = n, k - 1
        if self.binom(top, k) > idx:
            count = top - bottom
            while count > 0:
                step = count // 2
                mid = top - step
                if self.binom(mid, k) > idx:
                    top = mid - 1
                    count -= step + 1
                else:
                    count = step
        return top

    def get_simplex_vertices(self, idx, dim, n):
        """simplex: i_k < i_{k + 1}"""
        simplex = []
        n -= 1
        for k in range(dim + 1, 0, -1):
            n = self.get_max_vertex(idx, n, k)
            simplex.append(n)
            idx -= self.binom(n, k)
        return reversed(simplex)

    def get_cofacets(self, simplex: list[int], r: int = 0):
        """simplex: i_k < i_{k + 1}"""
        cofacets = []
        for k in range(self.n - 1, r - 1, -1):
            kix = bisect.bisect_left(simplex, k)
            if kix != len(simplex) and simplex[kix] == k:
                continue

            s = 0
            for l in range(kix):
                s += self.binom(simplex[l], l + 1)
            s += self.binom(k, kix + 1)
            for l in range(kix, len(simplex)):
                s += self.binom(simplex[l], l + 2)
            cofacets.append(s)
        return cofacets

    def get_facetes(self, simplex: list[int]):
        """simplex: i_k < i_{k + 1}"""
        facets = []
        s = 0
        for l in range(len(simplex) - 1):
            s += self.binom(simplex[l], l + 1)
        facets.append(s)
        for k in range(len(simplex) - 2, -1, -1):
            s -= self.binom(simplex[k], k + 1)
            s += self.binom(simplex[k + 1], k + 1)
            facets.append(s)
        return facets

    def assemble_columns_to_reduce(self, simplexes: list[Simplex], dim):
        res = []
        for s in simplexes:
            res.extend(
                self.get_cofacets(
                    self.get_simplex_vertices(s.idx, dim, dim + 1),
                    self.get_max_vertex(s.idx, dim, dim + 1)
                )
            )
        return res

    def get_zero_apparent_cofacet(self, simplex):
        cofacets = self.get_cofacets(simplex)

    def is_in_zero_apparent_pair(self, simplex):
        c = self.get_zero_apparent_cofacet(simplex)
        facets = self.get_facetes(simplex)

    @staticmethod
    def low(b: torch.Tensor, j: int):
        """
        we define low(j) to be the largest index value i such that x[i][j] is different from 0
        """
        i = -1
        for k in range(b.shape[0]):
            if b[k][j] != 0:
                i = k
        return None if i == -1 else i

    @staticmethod
    def extract(b: torch.Tensor, dg: list, diagram: dict):
        """
        :param b: Boundary matrix
        :param dg: When each simplex first appeared
        :param diagram: Persistence diagram
        :return:
        """

        for j in range(b.shape[0]):
            i = VietorisRipsComplex.low(b, j)
            if i is not None:
                # \sigma_i -> \sigma_j
                diagram[(i, j)] = dg[i], dg[j]
            else:
                unpaired = True
                for k in range(b.shape[0]):
                    if VietorisRipsComplex.low(b, k) == j:
                        # \sigma_j -> \sigma_k
                        unpaired = False
                        diagram[(j, k)] = dg[j], dg[k]
                if unpaired:
                    pass  # Don't keep infinite

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            return [self.filtration(x[0]), self.filtration(x[1]), self.filtration(x[2])]
        else:
            return self.filtration(x)

    def filtration(self, x: torch.Tensor):
        ...
