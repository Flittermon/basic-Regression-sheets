# LINEAR REGRESSION MODEL

import numpy as np
#dataset:

#create a linear regression model
def my_linear_model(x,w,b):
    """
        input:
        x = matrix of features
        w, b = parameters
        
        output:
        y = vector of predictions
    """
    
    return np.dot(x,w) + b

#create a logistic regression model
def my_logistic_model(x,w,b):
    """
        input:
        x = matrix of features
        w, b = parameters
        
        output:
        y = vector of predictions
    """
    
    prediction = 1 / (1+ np.exp(-(np.dot(x,w)+b)))
    return prediction

#compute the squared cost for our linear regression model
def linear_cost_function(x, y ,w ,b, lambda_ = 0):
    """

    Args:
        x (nparray): matrix with features
        y (nparray): vector with targets
        w (nparray): parameter
        b (float): parameter
        lambda_ (int, optional): default 0, use for normalization

    Returns:
        squared cost
    """
    
    y_pred = my_linear_model(x, w, b)
    cost = (y_pred - y) ** 2
    if lambda_ == 0:
        return np.sum(cost) / (2 * x.shape[0])
    else: 
        return np.sum(cost) / (2 * x.shape[0]) + (lambda_ / (2 * x.shape[0])) * np.sum(w ** 2)

#compute the loss for our logistic regression model
def logistic_cost_function(x, y, w, b, lambda_ = 0):
    """

    Args:
        x (nparray): matrix with features
        y (nparray): vector with targets
        w (nparray): parameter
        b (float): parameter
        lambda_ (int, optional): default 0, use for normalization

    Returns:
        loss
    """
    
    y_pred = my_logistic_model(x, w, b)
    
    loss = -y * np.log(y_pred) - (np.ones(shape= y_pred.shape)-y) * np.log(np.ones(shape= y_pred.shape) - y_pred)
    if lambda_ == 0:
        return np.sum(loss) / x.shape[0]
    else: 
        return np.sum(loss) / x.shape[0] + (lambda_ / (2 * x.shape[0])) * np.sum(w ** 2)

def compute_gradient_linear(x, y, w, b, lambda_ = 0):
    """

    Args:
        x (nparray): matrix with features
        y (nparray): vector with targets
        w (nparray): parameter
        b (float): parameter
        lambda_ (int, optional): default 0, use for normalization

    Returns:
        dj_dw, dj_db -> updated parameters to fit the model
    """
    
    m = x.shape[0]
    dj_dw, dj_db = 0, 0
    
    y_pred = my_linear_model(x, w, b)
    dj_dw = np.dot(x,(y_pred - y))
    dj_db = np.sum(y_pred - y)
    if lambda_ == 0:
        return dj_dw / m, dj_db/m
    else:
        return (dj_dw + (lambda_ * w)) / m, dj_db / m

def compute_gradient_logistic(x, y, w, b, lambda_ = 0):
    """

    Args:
        x (nparray): matrix with features
        y (nparray): vector with targets
        w (nparray): parameter
        b (float): parameter
        lambda_ (int, optional): default 0, use for normalization

    Returns:
        dj_dw, dj_db -> updated parameters to fit the model
    """
    
    m = x.shape[0]
    y_pred = my_logistic_model(x, w, b)
    
    dj_dw = np.dot(x,(y_pred - y))
    dj_db = np.sum(y_pred - y)
    
    if lambda_ == 0:
        return dj_dw / m, dj_db/m
    else:
        return (dj_dw + (lambda_ * w)) / m, dj_db / m
    
def batch_gradient_descent_linear(x, y, w_in, b_in, alpha, num_iters, lambda_ = 0):
    """_summary_

    Args:
        x (nparray): matrix with features
        y (nparray): vector with targets
        w_in (nparray): parameter
        b_in (float): parameter
        alpha (float): learning rate 
        num_iters (int): number of iterations
        lambda_ (int, optional): default 0, use for normalization

    Returns:
        new_w, new_b -> updated parameters over many iterations with learning rate alpha 
    """
    
    new_w = w_in
    new_b = b_in
    if lambda_ == 0:
        
        for i in range(num_iters):
            
            dj_dw, dj_db = compute_gradient_linear(x, y, new_w, new_b)
            new_w -= alpha * dj_dw
            new_b -= alpha * dj_db
            
            if i% (num_iters/10) == 0:
                print(f"Iteration number {i}")
                print(f"cost: {linear_cost_function(x, y, new_w, new_b)}, w: {new_w}, b: {new_b}")    
                
    else: 
        for i in range(num_iters):
            
            dj_dw, dj_db = compute_gradient_linear(x, y, new_w, new_b, lambda_)
            new_w -= alpha * dj_dw
            new_b -= alpha * dj_db
            
            if i% (num_iters/10) == 0:
                print(f"Iteration number {i}")
                print(f"cost: {linear_cost_function(x, y, new_w, new_b)}, w: {new_w}, b: {new_b}")
                
    return new_w, new_b

def batch_gradient_descent_logistic(x, y, w_in, b_in, alpha, num_iters, lambda_ = 0):
    """_summary_

    Args:
        x (nparray): matrix with features
        y (nparray): vector with targets
        w_in (nparray): parameter
        b_in (float): parameter
        alpha (float): learning rate 
        num_iters (int): number of iterations
        lambda_ (int, optional): default 0, use for normalization

    Returns:
        new_w, new_b -> updated parameters over many iterations with learning rate alpha 
    """
    
    new_w = w_in
    new_b = b_in
    if lambda_ == 0:
        
        for i in range(num_iters):
            
            dj_dw, dj_db = compute_gradient_logistic(x, y, new_w, new_b)
            new_w -= alpha * dj_dw
            new_b -= alpha * dj_db
            
            if i% (num_iters/10) == 0:
                print(f"Iteration number {i}")
                print(f"cost: {logistic_cost_function(x, y, new_w, new_b)}, w: {new_w}, b: {new_b}")    
                
    else: 
        for i in range(num_iters):
            
            dj_dw, dj_db = compute_gradient_logistic(x, y, new_w, new_b, lambda_)
            new_w -= alpha * dj_dw
            new_b -= alpha * dj_db
            
            if i% (num_iters/10) == 0:
                print(f"Iteration number {i}")
                print(f"cost: {logistic_cost_function(x, y, new_w, new_b)}, w: {new_w}, b: {new_b}")
                
    return new_w, new_b