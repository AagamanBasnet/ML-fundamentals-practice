#LOcally weighted regression
import numpy as np

def compute_weights(X,query_point,tau):
    m=X.shape[0]
    weights=np.zeros(m)
    for i in range(m):
        diff=X[i]-query_point
        weights[i]=np.exp(-np.sum(diff**2)/(2*tau**2))
    return np.diag(weights)
    
def locally_weighted_regression(X,y,query_point,tau):
    W=compute_weights(X,query_point,tau)
    theta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y


    prediction=query_point@theta
    return prediction

def lwr_predict_multiple(X_train,y_train,X_test,tau):
    predictions=[]

    for query_point in X_test:
        pred=locally_weighted_regression(X_train,y_train,query_point,tau)
        predictions.append(pred)
    
    return np.array(predictions)

np.random.seed(42)
X_lwr = np.linspace(0, 10, 50).reshape(-1, 1)
y_lwr = np.sin(X_lwr).flatten() + np.random.normal(0, 0.1, 50)

# Add bias term
X_lwr_bias = np.c_[np.ones(X_lwr.shape[0]), X_lwr]

# Test points
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]

# Predict with different tau values
tau = 0.5
predictions = lwr_predict_multiple(X_lwr_bias, y_lwr, X_test_bias, tau)

print(f"Tau (bandwidth) = {tau}")
print(f"Made predictions for {len(X_test)} test points")

# ============================================
# EXAMPLE 2: LOGISTIC REGRESSION (GRADIENT DESCENT)
# ============================================
print("LOGISTIC REGRESSION (GRADIENT DESCENT)")
print("="*50)



