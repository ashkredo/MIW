import pandas as pd
import numpy as np
from numpy import sin

import matplotlib.pylab as plt

def main():
#    data = pd.read_csv('data/dane2b.txt', header=None)
#    print(data)
    data = None
    with open('dane2.txt', 'r') as plik:
        data = np.loadtxt(plik)



    X = data[:,0]
    Y = data[:,1]
    X = X.reshape(61,1)
    Y = Y.reshape(61,1)

    c = np.hstack([X*X,X, np.ones(X.shape)])
    v = np.linalg.pinv(c) @ Y

    #print('v shape = {}'.format(v.shape))
    #print(v)

    plt.plot(X, Y)
    plt.plot(X, v[0]*X*X + v[1]*X+v[2])
    Y_prim = v[0]*X*X + v[1]*X+v[2]
    error = Y - Y_prim
    error = error*error
    print(error.mean())
    plt.show()



    x = data[:,0]
    y = data[:,1]
    x = x.reshape(61,1)
    y = y.reshape(61,1)

    c = np.hstack([np.power(x,4), np.power(x,3), np.power(x,2), x, np.ones(x.shape)])
    v = np.linalg.pinv(c) @ y

    #print('v shape = {}'.format(v.shape))
    #print(v)

    plt.plot(x, y)
    plt.plot(X, v[0]*np.power(x,4) +v[1]*np.power(x,3) + v[2]*np.power(x,2) + v[3]*X+v[4])
    Y_prim = v[0]*np.power(x,4) +v[1]*np.power(x,3) + v[2]*np.power(x,2) + v[3]*X+v[4]
    error = Y - Y_prim
    error = error*error
    print(error.mean())
    plt.show()


   


if __name__ == '__main__':
    main()    