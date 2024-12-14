import numpy as np
import math


class Model:

    def __init__(self, N, J, T, h=0, trails=1000):
        self.N = N
        self.model = np.random.randint(2, size=(self.N, self.N))
        self.J = J
        self.T = T
        self.h = h
        self.comparisonsMatch = []
        self.runs = 0
        self.comparisonsDiff = []
        self.similarityRatio = []
        self.upCountList = np.zeros(trails)
        self.downCountList = np.zeros(trails)
        for i in range(N):
            for j in range(N):
                if self.model[i, j] == 0:
                    self.model[i, j] = -1

    def get(self):
        return self.model

    def reset(self, T):
        self.T = T
        self.model = np.random.randint(2, size=(self.N, self.N))
        self.runs = 0
        for i in range(self.N):
            for j in range(self.N):
                if self.model[i, j] == 0:
                    self.model[i, j] = -1

    def compareTime(self, initialModel):
        match = 0
        difference = 0
        for i in range(self.N):
            for j in range(self.N):
                if initialModel[i, j] == self.model[i, j]:
                    match += 1
                elif initialModel[i, j] != self.model[i, j]:
                    difference += 1
        self.comparisonsMatch.append(float(match) / self.N ** 2)
        self.comparisonsDiff.append(float(difference) / self.N ** 2)
        if match == 0 or difference == 0:
            self.similarityRatio.append(0)
        else:
            self.similarityRatio.append(match / difference)

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

        return -1 * self.J * (horiz + vert) - self.h * np.sum(self.model)  # accounts for external magnetic field

    def updatePoint(self):
        r = np.random.randint(0, self.N)
        c = np.random.randint(0, self.N)
        epsilon = 0

        if r + 1 == self.N and c + 1 != self.N and c != 0:
            epsilon = 4 * self.J * self.model[r, c] * (
                    self.model[0, c] + self.model[r - 1, c] + self.model[r, c + 1] +
                    self.model[r, c - 1])
        elif r == 0 and c + 1 != self.N and c != 0:
            epsilon = 4 * self.J * self.model[r, c] * (
                    self.model[r + 1, c] + self.model[self.N - 1, c] +
                    self.model[r, c + 1] + self.model[r, c - 1])
        elif c + 1 == self.N and r + 1 != self.N and r != 0:
            epsilon = 4 * self.J * self.model[r, c] * (
                    self.model[r + 1, c] + self.model[r - 1, c] + self.model[r, 0] +
                    self.model[r, c - 1])
        elif c == 0 and r + 1 != self.N and r != 0:
            epsilon = 4 * self.J * self.model[r, c] * (
                    self.model[r + 1, c] + self.model[r - 1, c] + self.model[r, c + 1] +
                    self.model[r, self.N - 1])
        elif r + 1 == self.N and c == 0:
            epsilon = 4 * self.J * self.model[r, c] * (
                    self.model[0, c] + self.model[r - 1, c] + self.model[r, c + 1] +
                    self.model[r, self.N - 1])
        elif r + 1 == self.N and c + 1 == self.N:
            epsilon = 4 * self.J * self.model[r, c] * (
                    self.model[0, c] + self.model[r - 1, c] + self.model[r, 0] +
                    self.model[r, c - 1])
        elif r == 0 and c + 1 == self.N:
            epsilon = 4 * self.J * self.model[r, c] * (
                    self.model[r + 1, c] + self.model[self.N - 1, c] + self.model[r, 0] +
                    self.model[r, c - 1])
        else:
            epsilon = 4 * self.J * self.model[r, c] * (
                    self.model[r + 1, c] + self.model[r - 1, c] + self.model[r, c + 1] +
                    self.model[r, c - 1])

        epsilon += self.h * self.model[r, c]  # accounts for external magnetic field

        if epsilon < 0:
            self.model[r, c] = -self.model[r, c]
        else:
            p = math.exp(-1 * epsilon / self.T)
            if np.random.rand() <= p:
                self.model[r, c] = -self.model[r, c]
