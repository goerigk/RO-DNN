# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 19:10:51 2020

@author: Jannis
"""


import ROfeas
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import random as r
import gurobipy as gp
from gurobipy import GRB
import csv

import pandas


datadir="../src/selectiondata16/"
resultdir="../src/results16/"

csvFile = open("outputfeas.csv", 'w')
out = csv.writer(csvFile,delimiter=",", lineterminator = "\n")
csvFile.close()


for A in [1]:
        res_kern_obj = []
        res_kern_feas = []
        res_nn_obj = []
        res_nn_feas = []
        for C in range(1,11,1):
            print("A C", A, C)
            t_res_kern_obj = []
            t_res_kern_feas = []
            t_res_nn_obj = []
            t_res_nn_feas = []
            
            
            fileName = datadir+"test-"+str(A)+"-"+str(C)+".txt"
            X_test = np.genfromtxt(fileName, delimiter=',')
            fileName = datadir+"train-"+str(A)+"-"+str(C)+".txt"
            X_train = np.genfromtxt(fileName, delimiter=',')
            
            N=X_train.shape[1]
            
            RHS = 0            
            ALLX = []
            ALLT = []
            
            for quantil in [0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]:
                
                #solve kernel
                #Q,theta,alphas,SV
                Q = np.array(pandas.read_csv(resultdir+"q-"+str(A)+"-"+str(C)+"-"+("%.2f" % quantil)+".txt", header=None))
                theta = np.array(pandas.read_csv(resultdir+"theta-"+str(A)+"-"+str(C)+"-"+("%.2f" % quantil)+".txt", header=None))[0]
                SV = np.array(pandas.read_csv(resultdir+"sv-"+str(A)+"-"+str(C)+"-"+("%.2f" % quantil)+".txt", header=None))[0]
                alphas = np.array(pandas.read_csv(resultdir+"alphas-"+str(A)+"-"+str(C)+"-"+("%.2f" % quantil)+".txt", header=None))[0]
                
                start = time.time()
                objKern, x_kern = ROfeas.solveKernel(N,X_train,Q,theta,alphas,SV,RHS)
                end = time.time()
                t_k = end-start
                ALLX.append(x_kern)
                ALLT.append(t_k)
                
                avgkern, maximumkern, valskern = ROfeas.evaluateSolution(x_kern,X_test)
                
                t_res_kern_obj.append(sum(x_kern))
                t_res_kern_feas.append(sum([1 for i in range(len(valskern)) if valskern[i] <= RHS+0.01])/len(valskern))
                
            print(t_res_kern_obj)
            print(t_res_kern_feas)
                
            #solve neural network
            
            for quantil in [0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]:
                E = 3
                L = 3
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
                   
                
                    
                R,sigmas,lb,ub = ROfeas.getRadiiDataPoints(L,c0,X_train,listW, 0, quantil)
                    
                    
                start = time.time()
                obj, x = ROfeas.solveRobustSelection(N,L,dimLayers,c0, R, listW, M, lb, ub, sigmas,RHS)
                end = time.time()
                t = end-start
                
                ALLX.append(x)
                ALLT.append(t)
                
                    
                avg, maximum, vals = ROfeas.evaluateSolution(x,X_test)
                                
                t_res_nn_obj.append(sum(x))
                t_res_nn_feas.append(sum([1 for i in range(len(vals)) if vals[i] <= RHS+0.01])/len(vals))
                    
                        
                
            
            plt.plot(t_res_kern_obj,t_res_kern_feas,"-bo", fillstyle = "none", label="Kernel")
            plt.plot(t_res_nn_obj,t_res_nn_feas,":ro",  fillstyle = "none", label="NN")

            plt.legend(loc="best")
            plt.xlabel("Objective Value")
            plt.ylabel("Protection Level")
            plt.savefig("Histogramme/feas-" + str(A)+"-"+str(C)+".pdf")
            plt.clf()
            
            csvFile = open("outputfeas.csv", 'a')
            out = csv.writer(csvFile,delimiter=",", lineterminator = "\n")
            row = [A,C]
            
            for i in range(len(ALLT)):
                row.append(float(ALLT[i]))
            
            for i in range(len(ALLX)):
                for j in range(len(ALLX[i])):
                    row.append(float(ALLX[i][j]))
                    
            row.append(t_res_kern_obj)
            row.append(t_res_kern_feas)
            row.append(t_res_nn_obj)
            row.append(t_res_nn_feas)
            
            out.writerow(row)
            csvFile.close()

