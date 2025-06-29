from matrix_ops import * 
import numpy as np 
import random 
import pandas as pd 
from pprint import pprint as print 


def PCA(X, k=2):
    """
    Computes All Principal Components and its variance capture using SVD method
    Steps:
        * Compute S,sigma,V.T 
        * extract PCA from V.T.T
        * S gives relative ranking of 
    """
    # standardise data 
    # X shape (m,n)
    X = (X - np.mean(X,axis=0))/ np.std(X,axis=0)
    
    S,sigma,VT = np.linalg.svd(X) # (m,m) , min(m, n), (n,n)

    print((
        "S",
        S,
        S.shape,
        "sigma",
        sigma,
        sigma.shape,
        "V",
        VT,
        VT.shape
    ))
    eigen_values = sigma**2/(len(X) - 1)

    print((f"Eigen Values : ",eigen_values))
    explained_variance = eigen_values / np.sum(eigen_values)
    
    print(("Eigen Values Explained Varaince : ", explained_variance))

    eigen_vectors = VT.T # shape would be after transpose (n,n) -> (n,n)
    
    # top k components with highest variance are

    X_pca = X@eigen_vectors

    print(("Transformed Data : ", X_pca, X_pca.shape ))
    

    pca_loadings = eigen_vectors[:,:k] @ np.diag( np.sqrt(eigen_values[:k]) )

    print(("PCA Loadings", pca_loadings))
    return X_pca[:,:k]


if __name__ == "__main__":
    dummy_X = np.random.rand(5,10)
    
    print(dummy_X)
    
    PCA(dummy_X)