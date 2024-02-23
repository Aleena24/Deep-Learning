import numpy as np
import matplotlib.pyplot as plt

# Define the functions
def f1(x):
    return x**2 - 2*x + 2

def f2(x, y):
    return (1 - x)**2 + 100*(y - x**2)**2

# Gradient descent algorithm
def gradient_descent(f, grad_f, initial_point, learning_rate=0.01, epsilon=1e-5, max_iterations=1000):
    current_point = initial_point
    iterations = 0
    path = [initial_point]
    
    while True:
        gradient = grad_f(*current_point)
        new_point = tuple(current_point[i] - learning_rate * gradient[i] for i in range(len(current_point)))
        if np.linalg.norm(np.array(new_point) - np.array(current_point)) < epsilon or iterations >= max_iterations:
            break
        current_point = new_point
        iterations += 1
        path.append(new_point)
        
    return current_point, iterations, path

# Gradient of the functions
def grad_f1(x):
    return 2*x - 2
    
def grad_f2(x, y):
    df_dx = 2*(200*x**3 - 200*x*y + x - 1)
    df_dy = 200*(y - x**2)
    return df_dx, df_dy

# Function to plot the optimization path
def plot_path(path, ax):
    x_values = [point[0] for point in path]
    y_values = [point[1] for point in path]
    ax.plot(x_values, y_values, 'ro-')

# Optimize function f1(x) = x^2 - 2x + 2
initial_point_f1 = 0  # Initial guess
optimal_point_f1, iterations_f1, path_f1 = gradient_descent(f1, grad_f1, initial_point_f1)
print("Optimal point for f1(x):", optimal_point_f1)
print("Number of iterations for f1(x):", iterations_f1)

# Optimize function f2(x, y) = (1 - x)^2 + 100(y - x^2)^2
initial_point_f2 = (-2, 2)  # Initial guess
optimal_point_f2, iterations_f2, path_f2 = gradient_descent(f2, grad_f2, initial_point_f2)
print("Optimal point for f2(x, y):", optimal_point_f2)
print("Number of iterations for f2(x, y):", iterations_f2)

# Plotting
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f2(X, Y)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot for f1(x) = x^2 - 2x + 2
axs[0].plot(x, f1(x), label='f(x) = x^2 - 2x + 2')
axs[0].set_title('Optimization of f(x)')
plot_path(path_f1, axs[0])
axs[0].legend()

# Plot for f2(x, y) = (1 - x)^2 + 100(y - x^2)^2
axs[1].contour(X, Y, Z, levels=50, cmap='viridis')
axs[1].plot(optimal_point_f2[0], optimal_point_f2[1], 'ro', label='Optimal Point')
axs[1].set_title('Optimization of f(x, y)')
plot_path(path_f2, axs[1])
axs[1].legend()

plt.show()
