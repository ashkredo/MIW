# Regresja kwadratowa

import matplotlib.pyplot as plt
import numpy as np

def main():
    sharp = None
    with open ('D:\Sharp_char.txt', 'r') as plik:
        sharp = np.loadtxt(plik)
    print(sharp)

    x = sharp[:, 1].reshape(3, 1)
    y = sharp[:, 1].reshape(3, 1)

if __name__=='__main__':
    main()