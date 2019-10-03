# PLOT

import matplotlib.pyplot as plt
import numpy as np

def main():
    y = [123,53,12,76,2]

    plt.plot(y)
    plt.show()

    plt.plot(y, linestyle='--', color='y')
    plt.ylabel('Wartosci z listy y')
    plt.xlabel('Wartosci z listy x')
    plt.show()

    x = np.linspace(5,20,40)
    y = np.linsin(x)*x
    noise = np.random.normal(size=y.shape, scale=5)
    plt.plot(y, linestyle='--', marker='+', color='y', label='Sygnal wiesciowy')
    plt.plot(y+noise, linestyle='-', color='r', label='Sygnal wiesciowy')
    plt.ylabel('Sila sygnalu')
    plt.xlabel('Chas')
    plt.legend()
    ptl.show()

if __name__=='__main__':
    main()