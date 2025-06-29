import random


def generate_random_matrix(shape) :
    return [[random.random() for i in range(shape[1])] for j in range(shape[0])]

def matrix_shape(A):
    return (len(A), len(A[0]))

def matrix_dot_product(A, B) :
    assert matrix_shape(A)[1] == matrix_shape(B)[0], "Shape Mis-Match Matrix Product not Possible"
    
    result = []
    for i in range(matrix_shape(A)[0]):
        c = []
        for j in range(matrix_shape(B)[1]):
            # pos in result is (i,j)
            clm_a = [ii[i] for ii in A ]
            row_b = B[j]
            
            a = 0
            for ii in range(len(clm_a)):
                a+= clm_a[ii] * row_b[ii]
            
            c.append(a)
        
        result.append(c)
    
    return result


def transpose_matrix(A):
    shape = matrix_shape(A)
    
    transposed_A = []
    for i in range(shape[1]):
        n_r = []
        for j in range(shape[0]):
            n_r.append(A[j][i])
        transposed_A.append(n_r)

    return transposed_A

def matrix_element_iterator(A):
    for i in A:
        for j in i:
            yield j 

def reshape_matrix(A, new_shape):
    current_shape =  matrix_shape(A)
    if current_shape[0] * current_shape[1] != new_shape[0] * new_shape[1] :
        raise Exception("Matrix shape mismatch")
    
    matrix_elem_iter = iter(matrix_element_iterator(A))
    reshaped_matrix = []
    for i in range(new_shape[0]):
        c = []
        for j in range(new_shape[1]):
            c.append(next(matrix_elem_iter))
        
        reshaped_matrix.append(c)
    
    return reshaped_matrix

def mean(A, axis=0):
    dims = matrix_shape(A)
    mean_axis = [0] * dims[axis]
    
    for i in range(dims[0]):
        for j in range(dims[1]):
            mean_axis[(i,j)[axis]] += A[i][j]
    
    return [i/dims[axis] for i in mean_axis]
    

def scalar_multip(A,k):
    dims = matrix_shape(A)
    
    for i in range(dims[0]):
        for j in range(dims[1]):
            A[i][j] *= k 
    return A 


def det_matrix_cofactor_method(A):
    curr_shape = matrix_shape(A)
    if curr_shape[0] != curr_shape[1]:
        raise Exception("Non Square Matrix")
    
    if curr_shape[0] == curr_shape[1] == 2:
        return  A[0][0] * A[1][1] - A[0][1] * A[1][0]
    elif curr_shape[0] == curr_shape[1] == 1:
        return A[0][0]
     # row wise deter allways 
    deter_row = 0
     
    deter = 0 
    for deter_clm in range(curr_shape[1]):
        curr_cofactor = (-1) ** (deter_row+1 + deter_clm+1)
        subset_matrix = []
        for i in range(curr_shape[0]):
           r = []
           for j in range(curr_shape[1]):
               if i != deter_row and j != deter_clm:
                   r.append(A[i][j])

           if len(r) > 0: subset_matrix.append(r)
           
        
        deter += curr_cofactor * A[deter_row][deter_clm] * det_matrix_cofactor_method(subset_matrix)
    
    return deter
            
         
from numpy.linalg import det 
import numpy as np 

if __name__ == '__main__':
    A = generate_random_matrix((5,5))
    B = generate_random_matrix((5,4))
    
    print(
        A,
        '\n',
        det_matrix_cofactor_method(
            A
        )
    )
    
    print(
        det(
            np.array(A)
        )
    )