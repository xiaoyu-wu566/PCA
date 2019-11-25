
# coding: utf-8

# In[15]:


import numpy as np
import math
import csv
from scipy.optimize import minimize

t, x = [], []
with open("hw3_question2.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader)
    i = 0
    count = 0
    for row in csv_reader:
        count += int(row[1])
        if i % 24 == 0 and i != 0:
            x.append(count)
            t.append(i//24)
            count = 0
        i += 1

t, x = np.asarray(t), np.asarray(x)
t_train, x_train = t[:638], x[:638]
t_test, x_test = t[638:], x[638:]

# def lik(parameters):

nll = lambda theta: -np.sum(x*np.log(np.exp(theta[0]+t*theta[1]))-np.exp(theta[0]+t*theta[1]))

nll_model = minimize(nll, [20.0, 0.0], method='L-BFGS-B')
print(nll_model)
print('theta_0: {}; theta_1: {}'.format(nll_model['x'][0], nll_model['x'][1]))


'''
      fun: -18986446.016477134
 hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>
      jac: array([0.37252903, 5.58793545])
  message: b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
     nfev: 114
      nit: 22
   status: 0
  success: True
        x: array([5.84448203e+00, 4.67808684e-03])
[5.84448203e+00 4.67808684e-03]
'''
