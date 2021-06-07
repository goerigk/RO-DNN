import sys
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats
import sklearn.datasets

typ = int(sys.argv[1])
N = int(sys.argv[2])
noise = int(sys.argv[3])
seed = int(sys.argv[4])
SSIZE = int(sys.argv[5])

np.random.seed(seed=seed)

n=20

for i in range(noise):
    row = []
    for j in range(n):
        row.append(300*scipy.stats.uniform.rvs())
    print(row)


#multivariate normal
if typ == 1:
    vals = []
    Sigma = SSIZE*sklearn.datasets.make_spd_matrix(n_dim = n)
    mean = []
    for j in range(n):
        seg = scipy.stats.uniform.rvs()
        if seg <= 0.5:
            mean.append((-75 + 50*scipy.stats.uniform.rvs()))
        else:
            mean.append((25 + 50*scipy.stats.uniform.rvs()))
    for i in range(N):
        x = scipy.stats.multivariate_normal.rvs(mean = mean, cov = Sigma)
        vals.append(x)

    for i in range(N):
        row = []
        for j in range(n):
            row.append(max(0,vals[i][j]+150))
        print(row)


#mixed multivariate normal
if typ == 2:
    vals = []
    Sigma1 = SSIZE*sklearn.datasets.make_spd_matrix(n_dim = n)
    Sigma2 = SSIZE*sklearn.datasets.make_spd_matrix(n_dim = n)
    mean1 = np.zeros(n)
    mean2 = np.zeros(n)
    
    for j in range(n):
        seg = scipy.stats.uniform.rvs()
        if seg <= 0.5:
            mean1[j] = -75 + 50*scipy.stats.uniform.rvs()
        else:
            mean1[j] = 25 + 50*scipy.stats.uniform.rvs()
        
        seg = scipy.stats.uniform.rvs()
        if seg <= 0.5:
            mean2[j] = -75 + 50*scipy.stats.uniform.rvs()
        else:
            mean2[j] = 25 + 50*scipy.stats.uniform.rvs()
    
            
    for i in range(N):
        if scipy.stats.uniform.rvs() <= 0.5:
            x = scipy.stats.multivariate_normal.rvs(mean = mean1, cov = Sigma1)
            vals.append(x)
        else:
            x = scipy.stats.multivariate_normal.rvs(mean = mean2, cov = Sigma2)
            vals.append(x)

    for i in range(N):
        row = []
        for j in range(n):
            row.append(max(0,vals[i][j]+150))
        print(row)



#polyhedral (budgeted) uncertainty
if typ == 3:
    vals = np.zeros((N,n))
    lbs = []
    ubs = []
    for j in range(n):
        if scipy.stats.uniform.rvs() <= 0.5:
            v1 = -75 + 100*scipy.stats.uniform.rvs()
            v2 = -75 + 100*scipy.stats.uniform.rvs()
            lbs.append(min(v1,v2)-10)
            ubs.append(max(v1,v2)+10)
        else:
            v1 = 25 + 100*scipy.stats.uniform.rvs()
            v2 = 25 + 100*scipy.stats.uniform.rvs()
            lbs.append(min(v1,v2)-10)
            ubs.append(max(v1,v2)+10)
    
    Gamma = float(sys.argv[6])
    
    for i in range(N):
        gams = np.ones(n)
        sumg = 9999
        
        while sumg > Gamma:
            for j in range(n):
                gams[j] = scipy.stats.uniform.rvs()
            sumg = sum(gams)
    
        for j in range(n):
            vals[i][j] = lbs[j] + (ubs[j]-lbs[j])*gams[j]
    
    for i in range(N):
        row = []
        for j in range(n):
            row.append(max(0,vals[i][j]+150))
        print(row)
