import numpy as np

class Perceptron:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.learning_rate = 0.1

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels, epochs):
        for _ in range(epochs):
            errors = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                errors += int(error != 0)
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error
            print("Epoch:", _, " Errors:", errors)
            if errors == 0:
                break

# Define training data and labels for boolean functions
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels_and = np.array([0, 0, 0, 1])
labels_or = np.array([0, 1, 1, 1])
labels_nand = np.array([1, 1, 1, 0])
labels_xor = np.array([0, 1, 1, 0])

# Initialize perceptrons for each boolean function
and_perceptron = Perceptron(2)
or_perceptron = Perceptron(2)
nand_perceptron = Perceptron(2)
xor_perceptron = Perceptron(2)

# Train perceptrons
print("AND Function:")
and_perceptron.train(training_data, labels_and, epochs=10)
print("\nOR Function:")
or_perceptron.train(training_data, labels_or, epochs=10)
print("\nNAND Function:")
nand_perceptron.train(training_data, labels_nand, epochs=10)
print("\nXOR Function:")
xor_perceptron.train(training_data, labels_xor, epochs=10)
