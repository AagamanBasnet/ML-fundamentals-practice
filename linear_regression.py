import numpy as np

def initialize_parameters(no_of_features):
    theta=np.zeros(no_of_features)
    return theta

def predict(theta,X):
    return X @ theta

def compute_cost(X,y,theta):
    m=len(y)
    prediction=predict(theta,X)
    cost=(1/(2*m))*(np.sum((prediction-y)**2))
    return cost

def gradient_decent(X,y,theta,iterations,learning_rate):
    m=len(y)
    cost_history=[]

    for i in range(iterations):
        predictions=predict(theta,X)
        errors=predictions - y

        gradient=(1/m)*(X.T@errors)

        theta=theta-learning_rate*gradient
        cost=compute_cost(X,y,theta)
        cost_history.append(cost)

        if i % 200 == 0:
            print(f"Iteration {i}: Cost = {cost:.2f}, Theta = {theta}")

    return theta,cost_history


X=np.array([[600], [800], [1000], [1200]])
y=np.array([150, 200, 250, 300])

X=np.c_[np.ones(X.shape[0]),X]

theta=initialize_parameters(X.shape[1])
theta_final,cost=gradient_decent(
    X,y,theta,
    iterations=1000,
    learning_rate=0.0000001
)

print("final weights:",theta_final)
print("Final cost=",cost[-1])

new_house=np.array([[1400]])
new_house = np.c_[np.ones(new_house.shape[0]), new_house]
predicted_price=predict(theta_final,new_house)
print(f"predicted price:${predicted_price[0]:.1f}")        

