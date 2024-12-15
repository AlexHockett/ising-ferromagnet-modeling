import model
import matplotlib.pyplot as plt
import collective_model as cm
import numpy as np

default_N, default_J, default_T, default_h = 10, -1, 1, 0
default_trails = 1000

"""
N: Size of matrix (NxN) representing electron spins in ferromagnet
J: Proportionality constant in Hamiltonian
T: External temperature
h: Magnetic field parameter
trials: Number of iterations of the Monte Carlo algorithm
"""

use_custom_values = input("Do you want to input custom values (y/n):").lower().strip()

if use_custom_values == 'y':
    # Get custom inputs for N, J, T, h, and trials
    N = int(input("Enter the value for N: "))
    J = float(input("Enter the value for J: "))
    T = float(input("Enter the value for T: "))
    h = float(input("Enter the value for h: "))
    trails = int(input("Enter the value for number of Trails: "))
else:
    # Use default values
    N, J, T, h, trails = default_N, default_J, default_T, default_h, default_trails

clusterSize = 20
clusterChangeX = np.zeros(clusterSize)
for i in range(clusterSize):
    clusterChangeX[i] = i + 1

model = model.Model(N, J, T, h, trails)  
""" 
model takes parameters (N, J, T, h=0, trials)
see comment above for definitions of each variable
"""

print(model.get())
print(model.hamiltonian())
initialModel = np.copy(model.get())

for i in range(trails):  # Current number of trials = 1000, can be varied as desired
    model.updatePoint()
    model.compareTime(initialModel)
    # model.countSpins()

print(model.get())
print(model.hamiltonian())

cluster = cm.Collective_Model(10, -1, 1, h=1, trails=clusterSize)
print("Cluster Model: \n\n\n\n\n\n")
print(cluster.get())
print(cluster.hamiltonian())

for i in range(trails):  # Convergence is a lot faster for the cluster model
    cluster.createClusters()
    cluster.flip()
    cluster.reset_algorithm()

print(cluster.get())
print(cluster.hamiltonian())

Naxis = np.zeros(trails)
for i in range(len(Naxis)):
    Naxis[i] = i + 1

def graphCount(iterations, num, temperature):
    spinCountArray = np.zeros(iterations)
    model.reset(temperature)
    for j in range(num):
        for i in range(len(spinCountArray)):
            spinCountArray[i] = (spinCountArray[i] + model.upCountList[i])
            model.updatePoint()
            model.countSpins()
        model.reset(temperature)
    for i in range(len(spinCountArray)):
        spinCountArray[i] /= num
    return spinCountArray

def graphCountCM(iterations, num, temperature):
    spinCountArray = np.zeros(iterations)
    cluster.reset(temperature)
    for j in range(num):
        for i in range(len(spinCountArray)):
            spinCountArray[i] = (spinCountArray[i] + cluster.upCountList[i])
            cluster.createClusters()
            cluster.flip()
            cluster.reset_algorithm()
            cluster.countSpins()
        cluster.reset(temperature)
    for i in range(len(spinCountArray)):
        spinCountArray[i] /= num
    return spinCountArray

def graphHamiltonian(iterations, num, temperature):
    H = np.zeros(iterations)
    model.reset(temperature)
    for j in range(num):
        for i in range(len(H)):
            H[i] += model.hamiltonian()
            model.updatePoint()
        model.reset(temperature)
    for i in range(len(H)):
        H[i] /= num
    return H

plt.figure(1)
plt.plot(Naxis, graphHamiltonian(trails, 10, 1), 'r')
plt.plot(Naxis, graphHamiltonian(trails, 10, 3), 'b')
plt.plot(Naxis, graphHamiltonian(trails, 10, 5), 'y')
plt.plot(Naxis, graphHamiltonian(trails, 10, 7), 'g')
plt.show()

plt.figure(2)
graphChangeX = np.zeros(trails)
for i in range(trails):
    graphChangeX[i] = i + 1
plt.plot(graphChangeX, model.comparisonsMatch, 'r')
plt.plot(graphChangeX, model.comparisonsDiff, 'b')
plt.xlabel("Iterations")
plt.ylabel("Similarity")
plt.show()

plt.figure(3)
plt.plot(graphChangeX, model.similarityRatio, 'g')
plt.xlabel('Iterations')
plt.ylabel('Similarity to Differences Ratio')
plt.show()

plt.figure(4)
plt.plot(Naxis, graphCount(trails, 10, 1), 'b')
plt.plot(Naxis, model.upCountList, 'g')
plt.plot(Naxis, model.downCountList, 'r')
plt.xlabel('Iterations')
plt.ylabel('Spin Count')
plt.show()

plt.figure(5)
plt.plot(clusterChangeX, graphCountCM(clusterSize, 10, 1), 'b')
plt.plot(clusterChangeX, cluster.upCountList, 'g')
plt.plot(clusterChangeX, cluster.downCountList, 'r')
plt.xlabel('Iterations')
plt.ylabel('Spin Count')
plt.show()
