# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 19:10:51 2020

@author: Jannis
"""


import RO
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import random as r
import gurobipy as gp
from gurobipy import GRB
import csv

import pandas


datadir="../src/selectiondata14/"
resultdir="../src/results14/"


csvFile = open("outputvalues.csv", 'w')
out = csv.writer(csvFile,delimiter=",", lineterminator = "\n")
columnTitles = ["A","B","C","D","E","R-Quantil","t_d","t", "Obj_d", "Obj_NN", "TestAvg_d", "TestAvg_NN", "TestWC_d", "TestWC_NN", "FracBetter_d", "FracBetter_NN"]
out.writerow(columnTitles)
csvFile.close()

csvFile = open("outputsolutions.csv", 'w')
out = csv.writer(csvFile,delimiter=",", lineterminator = "\n")
csvFile.close()

for A in [1,2,3]:
        for C in range(21,31,1):
            for p in [0.3]:
                print("A C", A, C)
                fileName = datadir+"test-"+str(A)+"-"+str(C)+".txt"
                X_test = np.genfromtxt(fileName, delimiter=',')
                fileName = datadir+"train-"+str(A)+"-"+str(C)+".txt"
                X_train = np.genfromtxt(fileName, delimiter=',')
                
                N=X_train.shape[1]
                
                ALLX = []
                ALLT = []
                
                ALLAV = []
                ALLQUANT = []
                
                #solve kernel
                #Q,theta,alphas,SV
                Q = np.array(pandas.read_csv(resultdir+"q-"+str(A)+"-"+str(C)+"-0.90.txt", header=None))
                theta = np.array(pandas.read_csv(resultdir+"theta-"+str(A)+"-"+str(C)+"-0.90.txt", header=None))[0]
                SV = np.array(pandas.read_csv(resultdir+"sv-"+str(A)+"-"+str(C)+"-0.90.txt", header=None))[0]
                alphas = np.array(pandas.read_csv(resultdir+"alphas-"+str(A)+"-"+str(C)+"-0.90.txt", header=None))[0]
                
                start = time.time()
                objKern, x_kern = RO.solveKernel(p,N,X_train,Q,theta,alphas,SV)
                end = time.time()
                t_k = end-start
                ALLX.append(x_kern)
                ALLT.append(t_k)
                
                
                #solve neural network
                for E in [3]:
                    L = E
                    fileName = resultdir+"c-"+str(A)+"-"+str(C)+"-"+str(E)+".txt"
                    c0 = np.genfromtxt(fileName, delimiter=',')
                    
                    listW = []
                    dimLayers = []
                    
                    for F in range(0,L,1):
                        fileName = resultdir+"W-"+str(A)+"-"+str(C)+"-"+str(E)+"-"+str(F+1)+".txt"
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
                       

                    for quantil in [0.9]:
                                              
                        
                        R,sigmas,lb,ub = RO.getRadiiDataPoints(L,c0,X_train,listW, 0, quantil)
                        
                        
                        start = time.time()
                        obj, x = RO.solveRobustSelection(p,N,L,dimLayers,c0, R, listW, M, lb, ub, 0, sigmas=sigmas)
                        end = time.time()
                        t = end-start
                        
                        ALLX.append(x)
                        ALLT.append(t)
                        
                        
                        avg, maximum, vals = RO.evaluateSolution(x,X_test)
                        avgkern, maximumkern, valskern = RO.evaluateSolution(x_kern,X_test)
                        
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
                        plt.savefig("Histogramme/" + str(A)+"_"+str(C)+"_"+str(p)+".pdf")
                        plt.clf()
                        
                        ALLAV.append(sum(vals)/len(vals))
                        ALLAV.append(sum(valskern)/len(valskern))
                        
                        ALLQUANT.append(np.quantile(vals,quantil))
                        ALLQUANT.append(np.quantile(valskern,quantil))
                        
                        
                csvFile = open("outputsolutions.csv", 'a')
                out = csv.writer(csvFile,delimiter=",", lineterminator = "\n")
                row = [A,C]
                
                for i in range(len(ALLT)):
                    row.append(float(ALLT[i]))
                
                for i in range(len(ALLX)):
                    for j in range(len(ALLX[i])):
                        row.append(float(ALLX[i][j]))
                
                for i in range(len(ALLAV)):
                    row.append(ALLAV[i])
                
                for i in range(len(ALLQUANT)):
                    row.append(ALLQUANT[i])
                
                out.writerow(row)
                csvFile.close()

                    
