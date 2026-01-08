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

def gradient_descent_logistic(X,y,theta,iterations,learning_rate):
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

def newtons_method_logistic(X,y,theta,max_iterations=20,tol=1e-6):
    m=len(y)
    cost_history=[]

    for i in range(max_iterations):
        h=predict(theta,X)
        gradient=(1/m)*(X.T@(h-y))

        D=np.diag(h*(1-h))
        hessian=(1/m)*(X.T@D@X)

        hessian+=1e-8 * np.eye(hessian.shape[0])

        try:
            theta_update=np.linalg.solve(hessian,gradient)
            theta=theta-theta_update
        except np.linalg.LinAlgError:
            print("hessian is singular,stopping early")
            break

        cost=compute_logistic_cost(X,y,theta)
        cost_history.append(cost)

        print(f"Iteration{i} Cost={cost:.6f},Theta={theta}")

        if np.linalg.norm(theta_update) < tol:
            print (f"converged after{i+1} iterations")
            break

        return theta,cost_history


        

X=np.array([[1],[2],[3], [4], [6], [8]])
y=np.array([0,0,0, 0, 1, 1])

X=np.c_[np.ones(X.shape[0]),X]

print("=" * 60)
print("GRADIENT DESCENT")
print("=" * 60)
theta_gd = initialize_parameters(X.shape[1])
theta_gd, cost_gd = gradient_descent_logistic(
    X, y, theta_gd,
    iterations=1000,
    learning_rate=0.1
)
print(f"\nFinal weights (GD): {theta_gd}")
print(f"Final cost (GD): {cost_gd[-1]:.6f}")
print(f"Total iterations: 1000")

print("\n" + "=" * 60)
print("NEWTON'S METHOD")
print("=" * 60)
theta_newton = initialize_parameters(X.shape[1])
theta_newton, cost_newton = newtons_method_logistic(
    X, y, theta_newton,
    max_iterations=20
)
print(f"\nFinal weights (Newton): {theta_newton}")
print(f"Final cost (Newton): {cost_newton[-1]:.6f}")
print(f"Total iterations: {len(cost_newton)}")

print("\n" + "=" * 60)
print("PREDICTIONS")
print("=" * 60)
new_student = np.array([[1, 7]])

prob_gd = predict(theta_gd, new_student)
prob_newton = predict(theta_newton, new_student)

print(f"Student with {new_student[0,1]} hours:")
print(f"  GD prediction: {prob_gd[0]:.2%} chance of passing")
print(f"  Newton prediction: {prob_newton[0]:.2%} chance of passing")