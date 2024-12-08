import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Data Generation
def generate_data(num_samples=100, noise_level=10):
    np.random.seed(42)  # For reproducibility
    X = 2 * np.random.rand(num_samples, 1)
    y = 5 + 3 * X + noise_level * np.random.randn(num_samples, 1)
    return X, y

# Step 2: Add a Bias Term to the Input
def add_bias_term(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

# Step 3: Linear Regression Class
class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros((n, 1))  # Initialize weights
        for epoch in range(self.epochs):
            gradients = 2 / m * X.T.dot(X.dot(self.weights) - y)
            self.weights -= self.learning_rate * gradients

    def predict(self, X):
        return X.dot(self.weights)

    def get_weights(self):
        return self.weights

# Step 4: Main Script
if __name__ == "__main__":
    # Generate data
    X, y = generate_data()
    
    # Visualize data
    plt.scatter(X, y, color="blue", label="Data points")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.show()
    
    # Add bias term and split data
    X_bias = add_bias_term(X)
    X_train, X_test, y_train, y_test = train_test_split(X_bias, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LinearRegression(learning_rate=0.1, epochs=1000)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    # Visualize predictions
    plt.scatter(X_test[:, 1], y_test, color="blue", label="Actual")
    plt.scatter(X_test[:, 1], y_pred, color="red", label="Predicted")
    plt.plot(X_test[:, 1], y_pred, color="green", label="Regression line")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.show()
