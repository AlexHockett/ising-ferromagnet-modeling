import numpy as np
import math
from model import Model


class Collective_Model(Model):

  def __init__(self, N, J, T, h=0, trails=1000):
    super().__init__(N, J, T, h=h)
    self.adj = []
    self.spec = []
    self.vis2 = 0
    self.vis = np.zeros((self.N, self.N))
    self.upCountList = np.zeros(trails)
    self.downCountList = np.zeros(trails)
    self.runs = 0
    self.clust = []
    self.ghost = 0
    if h != 0:
      self.ghost = int(h / abs(h))
    for i in range(N):
      curr = []
      for j in range(N):
        curr.append([])
      self.adj.append(curr)

  def reset(self, T):
    self.T = T
    self.model = np.random.randint(2, size=(self.N, self.N))
    self.runs = 0
    for i in range(self.N):
      for j in range(self.N):
        if self.model[i, j] == 0:
          self.model[i, j] = -1

  def countSpins(self):
    upCount = 0
    downCount = 0
    for i in range(self.N):
      for j in range(self.N):
        if self.model[i, j] == 1:
          upCount += 1
        elif self.model[i, j] == -1:
          downCount += 1
    self.upCountList[self.runs] += upCount
    self.downCountList[self.runs] += downCount
    self.runs += 1

  def dfs(self, r, c, curr):
    if r == -1 and c == -1:
      if self.vis2:
        return
      self.vis2 = curr
      for x in self.spec:
        self.dfs(x[0], x[1], curr)
      self.clust[-1].append((r, c))
      return
    r = (r + self.N) % self.N
    c = (c + self.N) % self.N
    if self.vis[r][c]:
      return
    self.vis[r][c] = curr
    self.clust[-1].append((r, c))
    for x in self.adj[r][c]:
      self.dfs(x[0], x[1], curr)

  def createClusters(self):
    # derivation of probabilities: https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.58.86
    # https://www.sciencedirect.com/science/article/pii/0031891472900456
    # flood fill algorithm
    p = None
    if self.J > 0:
      p = 1 - math.exp(-2 * self.J / self.T)
      for r in range(self.N):
        for c in range(self.N):
          if r + 1 < self.N:
            if np.random.rand() <= p and self.model[r][c] == self.model[
                (r + 1) % self.N][c]:
              self.adj[r][c].append(((r + 1) % self.N, c))
            if np.random.rand() <= p and self.model[r][c] == self.model[r][
                (c + 1) % self.N]:
              self.adj[r][c].append((r, (c + 1) % self.N))
            if np.random.rand() <= p and self.model[r][c] == self.model[
                (r + self.N - 1) % self.N][c]:
              self.adj[r][c].append(((r + self.N - 1) % self.N, c))
            if np.random.rand() <= p and self.model[r][c] == self.model[r][
                (c + self.N - 1) % self.N]:
              self.adj[r][c].append((r, (c + self.N - 1) % self.N))
    else:
      p = 1 - math.exp(2 * self.J / self.T)
      for r in range(self.N):
        for c in range(self.N):
          if r + 1 < self.N:
            if np.random.rand() <= p and self.model[r][c] != self.model[
                (r + 1) % self.N][c]:
              self.adj[r][c].append(((r + 1) % self.N, c))
            if np.random.rand() <= p and self.model[r][c] != self.model[r][
                (c + 1) % self.N]:
              self.adj[r][c].append((r, (c + 1) % self.N))
            if np.random.rand() <= p and self.model[r][c] != self.model[
                (r + self.N - 1) % self.N][c]:
              self.adj[r][c].append(((r + self.N - 1) % self.N, c))
            if np.random.rand() <= p and self.model[r][c] != self.model[r][
                (c + self.N - 1) % self.N]:
              self.adj[r][c].append((r, (c + self.N - 1) % self.N))

    grp = 0
    for r in range(self.N):
      for c in range(self.N):
        p2 = 1 - math.exp(-abs(2 * self.h / self.T))
        if self.h < 0:
          if np.random.rand() <= p2 and self.ghost != self.model[r][c]:
            self.adj[r][c].append((-1, -1))
            self.spec.append((r, c))
        else:
          if np.random.rand() <= p2 and self.ghost == self.model[r][c]:
            self.adj[r][c].append((-1, -1))
            self.spec.append((r, c))

    for r in range(self.N):
      for c in range(self.N):
        if self.vis[r][c] == 0:
          grp += 1
          self.clust.append([])
          self.dfs(r, c, grp)

    if not self.vis2:
      grp += 1
      self.clust.append([])
      self.dfs(-1, -1, grp)

# if h > 0:
# ghost particle will be aligned with other particles in cluster

# if h < 0:
# ghost particle will be anti-aligned with other particles in cluster

# ghost particle can flip with rest of particles in cluster, but ghost particle IS NOT h

#(i.e. h = magnetic field coupling parameter)

  def flip(self):
    for x in self.clust:
      if np.random.rand() <= 0.5:
        for y in x:
          if y[0] == -1 and y[1] == -1:
            self.ghost = -1 * self.ghost
          else:
            self.model[y[0]][y[1]] = -1 * self.model[y[0]][y[1]]

  def reset_algorithm(self):
    self.vis = np.zeros((self.N, self.N))
    self.clust = []
    self.spec = []
    self.vis2 = 0
    for i in range(self.N):
      for j in range(self.N):
        self.adj[i][j] = []

  def hamiltonian(self):
    # following eq. (1)
    horiz = 0
    vert = 0
    for r in range(self.N):
      for c in range(self.N):
        if r + 1 == self.N and c + 1 != self.N:
          vert += self.model[r, c] * self.model[0, c]
          horiz += self.model[r, c] * self.model[r, c + 1]
        elif r + 1 != self.N and c + 1 == self.N:
          horiz += self.model[r, c] * self.model[r, 0]
          vert += self.model[r, c] * self.model[r + 1, c]
        elif r + 1 == self.N and c + 1 == self.N:
          horiz += self.model[r, c] * self.model[r, 0]
          vert += self.model[r, c] * self.model[0, c]
        else:
          horiz += self.model[r, c] * self.model[r, c + 1]
          vert += self.model[r, c] * self.model[r + 1, c]

    return -1 * self.J * (horiz + vert) - self.h * self.ghost * np.sum(
        self.model)
