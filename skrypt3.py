import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split


start = [ 'k' , 'p', 'n' ]
#prawdopodobienstwo s t a r tu
p_start = [ 0.3 , 0.5 , 0.2 ]
#macierz p r z e j s c i a
t1 = [  'k' , 'p', 'n'  ]
p_t1 =[ [ 0.53 , 0.23 , 0.24 ] , [ 0.3 , 0.1 , 0.6 ], [ 0.5 , 0.1 , 0.4 ] ]
t2 = [ "1" , "0" , "-1" ]
#macierz emi s j i
p_t2 =[ [ 0.4 , 0.4 , 0.2 ] , [ 0.33 , 0.47 , 0.2 ], [ 0.74 , 0.1 , 0.16 ] ]
initial = np.random.choice(start , replace=True , p=p_start )
n = 20
st = 1
for i in range (n ) :
    if st :
        state = initial
        st = 0
        print ( state )
    if state == 'k' :
        activity = np.random.choice( t2 , p=p_t2[0] )
        print ( state )
        print ( activity )
        state = np.random.choice( t1 , p=p_t1[0] )
    elif state == 'p' :
        activity = np.random.choice( t2 , p=p_t2[1] )
        print ( state )
        print ( activity )
        state = np.random.choice( t1 , p=p_t1[1] )
    elif state == 'n' :
        activity = np.random.choice( t2 , p=p_t2[2] )
        print( state )
        print ( activity )
        state = np.random.choice( t1 , p=p_t1[2] )
    print("\n")