import numpy as np

class SimplePerceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def activation(self, x):
        """Функция активации (пороговая функция)"""
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        """Обучение персептрона"""
        n_samples, n_features = X.shape
        
        # Инициализируем веса и смещение
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            # Прямой проход
            linear_output = np.dot(X, self.weights) + self.bias
            y_predicted = self.activation(linear_output)
            
            # Обновление весов
            update = self.learning_rate * (y - y_predicted)
            self.weights += np.dot(X.T, update)
            self.bias += self.learning_rate * np.sum(update)
    
    def predict(self, X):
        """Предсказание на основе обученной модели"""
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation(linear_output)
        return y_predicted

# Создаем тренировочную выборку
X = np.array([[0, 0],
              [0, 1],
              [1, 1]])
y = np.array([0, 1, 1])

# Создаем и обучаем персептрон
perceptron = SimplePerceptron(learning_rate=0.1, n_iterations=15)
perceptron.fit(X, y)

# Прогнозируем
predictions = perceptron.predict(np.array([[0, 0]]))
print("Предсказания:", predictions)
