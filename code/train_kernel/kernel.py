import numpy as np
import pandas
import sys

import scipy.stats

import cplex

print("Reading data.") 

data = np.transpose(np.array(pandas.read_csv(sys.argv[1], header=None)))

n=len(data)
N=len(data[0])

print ("n=",n)
print ("N=",N)




print("Calculating Eigenvalues.") 

#add whitenoise, only include if necessary
#for i in range(n):
    #for j in range(N):
        #data[i][j] = (0.001*(scipy.stats.uniform.rvs()-0.5)+1)*data[i][j]

A = np.cov(data)

eigVals, eigVecs = np.linalg.eig(A)

eigVals = eigVals**(-1/2)
eigVals = np.diag(eigVals)
Q = eigVecs.dot(eigVals).dot(eigVecs.T)





print("Creating kernel matrix.") 

l = np.zeros(n)
for i in range(n):
    l[i] = np.amax(Q[i].dot(data)) - np.amin(Q[i].dot(data)) + 0.01
lsum = np.sum(l)


#####prepare K values

v = []
for i in range(N):
    v.append(Q.dot(np.transpose(data)[i]))

K = [[lsum - np.linalg.norm(v[i]-v[j], ord=1) for j in range(N)] for i in range(N)]


nu = 1-float(sys.argv[2])

print(nu)





####solve qp

print("Solving QP.") 

p = cplex.Cplex()

ub = (1/(N*nu))*np.ones(N)
obj = np.zeros(N)
for i in range(N):
    obj[i] = -K[i][i]

varnames = ["alpha"+str(j) for j in range(N)]

p.variables.add(obj=obj, ub=ub, names = varnames)

p.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = range(N), val = [1.0]*N)], senses = ["E"], rhs = [1])


p.objective.set_sense(p.objective.sense.minimize)

ind = np.array([[x for x in range(N)] for x in range(N)])

qmat = []
for i in range(N):
    qmat.append([np.arange(N), K[i]])

p.objective.set_quadratic(qmat)

p.solve()




####calculate output

print("Calculating polyhedron.") 
eps=0.00001
alphas = [p.solution.get_values(i) for i in range(N)]
SV = [i for i in range(N) if alphas[i] > eps]
BSV = [i for i in SV if alphas[i] < 1/(N*nu)]

val = []
for i in BSV:
    vec = [alphas[j]*np.linalg.norm(Q.dot(np.transpose(data)[i]-np.transpose(data)[j]), ord=1) for j in SV]
    val.append(np.sum(vec))
theta = np.min(val)



###print results

np.set_printoptions(threshold=sys.maxsize)
print("SVBEGIN")
print(SV)
print("SVEND")

print("QBEGIN")
print(Q)
print("QEND")

print("THETABEGIN")
print(theta)
print("THETAEND")

print("ALPHASBEGIN")
print(alphas)
print("ALPHASEND")


print("thetaval", theta)
