# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 19:10:51 2020

@author: Jannis
"""


import ROsp
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import random as r
import gurobipy as gp
from gurobipy import GRB
import csv

import pandas


datadir="../data/graph-timesdata/"
resultdir="../src/spresults/"


##read graph

edges = np.genfromtxt(datadir+"graph.csv", delimiter=';', dtype='int')

n = np.amax(edges) + 1
m = len(edges)

print(n)
print(m)

##build neighbor list
inlist = []
outlist = []

for i in range(n):
    inlist.append([])
    outlist.append([])

for e in range(m):
    inlist[edges[e][1]].append(e)
    outlist[edges[e][0]].append(e)


pairs = [(34, 28), (52, 37), (41, 58), (53, 34), (17, 4), (21, 42), (7, 34), (29, 50), (29, 43), (11, 45)]



for A in ["weekends"]:
    
    for (startnode,stopnode) in pairs:
        ##scenarios
        ALLX = []
        ALLT = []
        ALLAV = []
        ALLQUANT = []
    
        
        X_test = np.genfromtxt(datadir+"test-"+A+".csv", delimiter=',')
        X_train = np.genfromtxt(datadir+"train-"+A+".csv", delimiter=',')
                
        #solve kernel
        #Q,theta,alphas,SV
        Q = np.array(pandas.read_csv(resultdir+"q-"+str(A)+".txt", header=None))
        theta = np.array(pandas.read_csv(resultdir+"theta-"+str(A)+".txt", header=None))[0]
        SV = np.array(pandas.read_csv(resultdir+"sv-"+str(A)+".txt", header=None))[0]
        alphas = np.array(pandas.read_csv(resultdir+"alphas-"+str(A)+".txt", header=None))[0]
        
        start = time.time()
        objKern, x_kern = ROsp.solveKernel(n,m,inlist,outlist,startnode,stopnode,X_train,Q,theta,alphas,SV)
        end = time.time()
        t_k = end-start
        ALLX.append(x_kern)
        ALLT.append(t_k)
        
        #solve neural network
        for E in [3]:
            L = E
            fileName = resultdir+"c-"+str(A)+".txt"
            c0 = np.genfromtxt(fileName, delimiter=',')
            
            listW = []
            dimLayers = []
            
            for F in range(0,L,1):
                fileName = resultdir+"W-"+str(A)+"-"+str(F+1)+".txt"
                listW.append(np.genfromtxt(fileName, delimiter=','))
                dimLayers.append(listW[F].shape[0])
                
            N=listW[0].shape[1]
            
            maxScenEntry = max(np.amax(X_train),np.amax(X_test))
            maxEntry = max(np.amax(X_train),np.amax(X_test))
            M=[]
            for i in range(0,L,1):
                rowSums = np.sum(np.absolute(listW[i]),axis=1)
                M.append(maxEntry*np.amax(rowSums))
                maxEntry = maxEntry*np.amax(rowSums)
               
            
            for quantil in [0.95]:
                                      
                R,sigmas,lb,ub = ROsp.getRadiiDataPoints(L,c0,X_train,listW, 0, quantil)
                
                
                start = time.time()
                obj, x = ROsp.solveRobustSelection(n,m,inlist,outlist,startnode,stopnode,L,dimLayers,c0, R, listW, M, lb, ub, 0, sigmas=sigmas)
                end = time.time()
                t = end-start
                
                ALLX.append(x)
                ALLT.append(t)
                
                avg, maximum, vals = ROsp.evaluateSolution(x,X_test)
                avgkern, maximumkern, valskern = ROsp.evaluateSolution(x_kern,X_test)
                left = min(min(vals),min(valskern))
                right = max(max(vals),max(valskern))
                
                f2,cx2 = np.histogram(vals,np.linspace(left, right, num=50))
                f3,cx3 = np.histogram(valskern,np.linspace(left, right, num=50))
                
                ymax = max(max(f2),max(f3))
                
                plt.plot(cx2[1:],f2,"-r", label="NN")
                plt.plot(cx3[1:],f3,"-b", label="Kernel")
                
                plt.vlines(np.quantile(vals,quantil),0,0.4*ymax,"r")
                plt.vlines(sum(vals)/len(vals),0,0.4*ymax,"r","dashed")
                
                plt.vlines(np.quantile(valskern,quantil),0,0.3*ymax,"b")
                plt.vlines(sum(valskern)/len(valskern),0,0.3*ymax,"b","dashed")

                plt.legend(loc="best")
                plt.xlabel("cx")
                plt.ylabel("freq")
                plt.savefig("Histogramme/" + str(A)+"-"+str(startnode)+"-"+str(stopnode)+".pdf")
                plt.clf()
                
                ALLAV.append(sum(vals)/len(vals))
                ALLAV.append(sum(valskern)/len(valskern))
                
                ALLQUANT.append(np.quantile(vals,quantil))
                ALLQUANT.append(np.quantile(valskern,quantil))
                
                
        csvFile = open("sp-solutions.csv", 'a')
        out = csv.writer(csvFile,delimiter=",", lineterminator = "\n")
        row = [A,startnode,stopnode]
        
        for i in range(len(ALLT)):
            row.append(float(ALLT[i]))
        
        for i in range(len(ALLX)):
            for j in range(len(ALLX[i])):
                row.append(float(ALLX[i][j]))
        
        for i in range(len(ALLAV)):
            row.append(ALLAV[i])
        
        for i in range(len(ALLQUANT)):
            row.append(ALLQUANT[i])
            
        for i in range(len(ALLX)):
            row.append(sum(ALLX[i]))
        
        out.writerow(row)
        csvFile.close()

                    
