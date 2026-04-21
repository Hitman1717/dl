class Perceptron:

    def __init__(self, learning_rate=0.1, epochs=10):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    # Activation function (Step function)
    def predict_single(self, x):
        summation = 0
        for i in range(len(x)):
            summation += self.weights[i] * x[i]
        summation += self.bias

        if summation >= 0:
            return 1
        else:
            return -1

    # Training function
    def fit(self, X, y):
        n_features = len(X[0])

        # Initialize weights and bias
        self.weights = []
        for i in range(n_features):
            self.weights.append(0)

        self.bias = 0

        # Training loop
        for epoch in range(self.epochs):
            for i in range(len(X)):
                x_i = X[i]
                target = y[i]

                prediction = self.predict_single(x_i)

                # Weight update
                if prediction != target:
                    for j in range(n_features):
                        self.weights[j] = self.weights[j] + self.lr * target * x_i[j]

                    # Bias update
                    self.bias = self.bias + self.lr * target

    # Predict multiple samples
    def predict(self, X):
        results = []
        for x in X:
            results.append(self.predict_single(x))
        return results

# ---------- MAIN PROGRAM ----------

if __name__ == "__main__":
    
    X = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    y = [-1, -1, -1, 1]

    model = Perceptron(learning_rate=0.1, epochs=10)
    
    model.fit(X, y)

    predictions = model.predict(X)

    print("Predictions:", predictions)