import numpy as np

def sigmoid(z):
    z=np.clip(z,-500,500)
    return 1/(1+np.exp(-z))

def initialize_parameters(no_of_features):
    theta=np.zeros(no_of_features)
    return theta

def predict(theta,X):
    return sigmoid(X @ theta)

def compute_logistic_cost(X,y,theta):
    m=len(y)
    h=predict(theta,X)
    epsilon=1e-15
    h=np.clip(h,epsilon,1-epsilon)

    cost=(-1/m)*np.sum(y*np.log(h)+(1-y)*np.log(1-h))
   
    return cost

def gradient_decent_logistic(X,y,theta,iterations,learning_rate):
    m=len(y)
    cost_history=[]

    for i in range(iterations):
        predictions=predict(theta,X)
        errors=predictions - y

        gradient=(1/m)*(X.T@errors)

        theta=theta-learning_rate*gradient
        cost=compute_logistic_cost(X,y,theta)
        cost_history.append(cost)

        if i % 200 == 0:
            print(f"Iteration {i}: Cost = {cost:.2f}, Theta = {theta}")

    return theta,cost_history

X=np.array([[1],[2],[3], [4], [6], [8]])
y=np.array([0,0,0, 0, 1, 1])

X=np.c_[np.ones(X.shape[0]),X]

theta=initialize_parameters(X.shape[1])
theta_logistic,cost=gradient_decent_logistic(
    X,y,theta,
    iterations=1000,
    learning_rate=0.1
)

print("final weights:",theta_logistic)
print("Final cost=",cost[-1])
new_student = np.array([[1, 7]])  # 5 hours of study
probability = predict(theta_logistic,new_student)
prediction = 1 if probability >= 0.5 else 0
print(f"Student with {new_student[0,1]} hours: {probability[0]:.2%} chance of passing")
print(f"Prediction: {'PASS' if prediction == 1 else 'FAIL'}")  

