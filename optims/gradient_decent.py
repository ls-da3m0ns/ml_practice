import numpy as np 

def mse(y,yh):
    return np.average((y-yh)**2)



def linear_regression_using_gradient_descent(X,y,num_steps=1000,lr=0.001, penalty='l2', lam = 0.5 ):
    """
    Uses Gradient Descent to optimize Linear Regression Model 
    For Element wise 
        yh = mx + c
        l = ((mx + c ) - y) ** 2 / len(y)

        dl / dm = 2*(mx + c - y) * x / len(y)
        dl / dc = 2*(mx + c - y) / len(y)

        m = m - alpha * dl/dm 
        c = c - alpha * dl/dc 
    """

    # Initialization 
    weight = np.random.rand(X.shape[1])
    bias = np.random.random()

    for epoch in range(num_steps):
        yh = X @ weight  + bias
        error = yh - y 


        dw = (1/len(X)) * X.T @ error 

        if penalty=='l1':
            dw = dw + lam * np.sign(weight)
        elif penalty == 'l2':
            dw  = dw + 2 * lam * weight

        db = (1/len(X)) * np.sum(error)

        weight = weight - lr*dw 
        bias = bias - lr*db 

        print(
            f"Epoch : {epoch}, MSE : {mse(y,yh)}"
        )




if __name__ == '__main__':
    X = np.random.rand(100,5)
    y = np.random.rand(100)

    linear_regression_using_gradient_descent(X,y)