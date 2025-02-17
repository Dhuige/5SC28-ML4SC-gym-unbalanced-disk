#%%
import numpy as np
import os
import matplotlib.pyplot as plt
# make sure that the current working directory is the directory of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
out = np.load('training-data.npz')
th_train = out['th'] #th[0],th[1],th[2],th[3],...
u_train = out['u'] #u[0],u[1],u[2],u[3],...

# data = np.load('test-prediction-submission-file.npz')
data = np.load('test-prediction-submission-file.npz')
upast_test = data['upast'] #N by u[k-15],u[k-14],...,u[k-1]
thpast_test = data['thpast'] #N by y[k-15],y[k-14],...,y[k-1]
# thpred = data['thnow'] #all zeros


def create_IO_data(u,y,na,nb):
    X = []
    Y = []
    for k in range(max(na,nb), len(y)):
        X.append(np.concatenate([u[k-nb:k],y[k-na:k]]))
        Y.append(y[k])
    return np.array(X), np.array(Y)

na = 5
nb = 5
Xtrain, Ytrain = create_IO_data(u_train, th_train, na, nb)

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(Xtrain,Ytrain)
#%%
Ytrain_pred = reg.predict(Xtrain)
print('train prediction errors:')
print('RMS:', np.mean((Ytrain_pred-Ytrain)**2)**0.5,'radians')
print('RMS:', np.mean((Ytrain_pred-Ytrain)**2)**0.5/(2*np.pi)*360,'degrees')
print('NRMS:', np.mean((Ytrain_pred-Ytrain)**2)**0.5/Ytrain.std()*100,'%')

 #only select the ones that are used in the example
Xtest = np.concatenate([upast_test[:,15-nb:], thpast_test[:,15-na:]],axis=1)

Ypredict = reg.predict(Xtest)

assert len(Ypredict)==len(upast_test), 'number of samples changed!!'
plt.plot(Ytrain_pred, c='r',alpha=0.5)
plt.scatter(np.arange(len(Ytrain)),Ytrain,alpha=1, s=0.1)
np.savez('test-prediction-example-submission-file.npz', upast=upast_test, thpast=thpast_test, thnow=Ypredict)
# %%
import numpy as np
import math
k = 1000
S = 1
Nsteps = 100
t = 1200
max_plottime = 6000  # seconds
Tstep = math.ceil(max_plottime / Nsteps)  # make the simulation as long as the measurement
time = np.arange(Nsteps) * Tstep  # Generate time values
time = time.astype(np.float64)
# Calculate x and y using the formula
x = time
y = np.square(time) + S * time/10000 + t

# Store x and y in an array
Mean_Result = np.column_stack((x, y))
print(y)
# %%
