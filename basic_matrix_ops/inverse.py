import scipy.linalg
from matrix_ops import * 
import numpy as np 
import random 
import pandas as pd 
import scipy
from pprint import pprint as print 

def matrix_inverse(X):
    """
    Computes inverse of square matrix using LU decompose 
        Finds Permutation, Lower and Upper triangular matrix
        # essentially trying to solve this at later 
        Ly = Pb
        Ux = y

        we know 
        PA = LU

    """
    assert X.shape[0] == X.shape[1], "Input Not a square matrix."
    assert np.linalg.det(X) != 0, "Zero Determinant of input"

    det_x = np.linalg.det(X)

    P,L,U = scipy.linalg.lu(X)

    I = np.eye(X.shape[0])

    X_inv = np.zeros(X.shape)

    # solving for each column independently 
    for i in range(X.shape[0]) :
        #Ly = Pb 
        y = scipy.linalg.solve(L, (P@I[:,i]))

        # Ux = y 
        x = scipy.linalg.solve(U, y)

        X_inv[:,i] = x 

    return X_inv



if __name__ == "__main__":
    dummy_X = np.random.rand(5,5)
    
    print(dummy_X)
    
    invese_X = matrix_inverse(dummy_X)
    print((
        "Inverse",
        invese_X
    ))