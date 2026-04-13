# import numpy as np
# import matplotlib.pyplot as plt

# # Функции активации
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def sigmoid_derivative(x):
#     return x * (1 - x)

# class NeuralNetwork:
#     def __init__(self, x, y, lr=0.1):
#         self.input = x
#         self.weights1 = np.random.randn(self.input.shape[1], 4) * 0.5  # инициализация с нормализацией
#         self.weights2 = np.random.randn(4, 1) * 0.5
#         self.y = y
#         self.output = np.zeros(self.y.shape)
#         self.lr = lr

#     def feedforward(self):
#         self.layer1 = sigmoid(np.dot(self.input, self.weights1))
#         self.output = sigmoid(np.dot(self.layer1, self.weights2))

#     def backprop(self):
#         # Производная функции потерь (MSE)
#         d_output = 2 * (self.y - self.output) * sigmoid_derivative(self.output)
        
#         # Градиенты
#         d_weights2 = np.dot(self.layer1.T, d_output)
#         d_weights1 = np.dot(self.input.T, np.dot(d_output, self.weights2.T) * sigmoid_derivative(self.layer1))
        
#         # Обновление весов
#         self.weights1 += self.lr * d_weights1
#         self.weights2 += self.lr * d_weights2
    
#     def predict(self, x):
#         """Предсказание для новых данных"""
#         layer1 = sigmoid(np.dot(x, self.weights1))
#         return sigmoid(np.dot(layer1, self.weights2))
    
#     def get_loss(self):
#         """Вычисление текущей потери (MSE)"""
#         return np.mean((self.y - self.output) ** 2)


# # ========== СОЗДАНИЕ ТЕСТОВЫХ ДАННЫХ ==========
# # Пример: XOR проблема (классическая задача для нейросети)
# np.random.seed(42)

# # XOR данные
# X = np.array([[0, 0],
#               [0, 1],
#               [1, 0],
#               [1, 1]])

# y = np.array([[0], 
#               [1], 
#               [1], 
#               [0]])

# print("Входные данные:")
# print(X)
# print("\nЦелевые значения:")
# print(y)

# # ========== ОБУЧЕНИЕ ==========
# # Создаем сеть
# nn = NeuralNetwork(X, y, lr=0.5)

# # Параметры обучения
# epochs = 10000
# losses = []  # список для хранения значений потерь

# print("\nНачинаем обучение...")
# print("=" * 50)

# # Цикл обучения
# for epoch in range(epochs):
#     # Прямое распространение
#     nn.feedforward()
    
#     # Сохраняем значение потерь
#     loss = nn.get_loss()
#     losses.append(loss)
    
#     # Обратное распространение
#     nn.backprop()
    
#     # Выводим информацию каждые 1000 эпох
#     if epoch % 1000 == 0:
#         print(f"Эпоха {epoch:5d}, Потери: {loss:.6f}")

# print("=" * 50)
# print("Обучение завершено!")

# # ========== РЕЗУЛЬТАТЫ ==========
# print("\nРезультаты после обучения:")
# print("-" * 50)
# for i in range(len(X)):
#     print(f"Вход: {X[i]} -> Предсказание: {nn.output[i][0]:.4f} (Ожидалось: {y[i][0]})")

# print(f"\nФинальные потери: {losses[-1]:.6f}")

# # ========== ПОСТРОЕНИЕ ГРАФИКА ==========
# plt.figure(figsize=(12, 5))

# # График 1: Уменьшение потерь
# plt.subplot(1, 2, 1)
# plt.plot(losses, 'b-', linewidth=1)
# plt.title('Уменьшение функции потерь (MSE)', fontsize=14, fontweight='bold')
# plt.xlabel('Эпоха', fontsize=12)
# plt.ylabel('Потери (MSE)', fontsize=12)
# plt.grid(True, alpha=0.3)
# plt.yscale('log')  # логарифмическая шкала для лучшей визуализации

# # Добавляем аннотацию с финальными значениями
# plt.text(0.95, 0.95, f'Финальные потери: {losses[-1]:.6f}', 
#          transform=plt.gca().transAxes, ha='right', va='top',
#          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# # График 2: Сравнение предсказаний с ожидаемыми
# plt.subplot(1, 2, 2)
# x_pos = np.arange(len(X))
# width = 0.35

# plt.bar(x_pos - width/2, y.flatten(), width, label='Ожидаемые', alpha=0.7, color='blue')
# plt.bar(x_pos + width/2, nn.output.flatten(), width, label='Предсказанные', alpha=0.7, color='orange')

# plt.xlabel('Пример', fontsize=12)
# plt.ylabel('Значение', fontsize=12)
# plt.title('Сравнение предсказаний с ожидаемыми значениями', fontsize=14, fontweight='bold')
# plt.xticks(x_pos, [f'{X[i]}' for i in range(len(X))])
# plt.legend()
# plt.grid(True, alpha=0.3, axis='y')

# plt.tight_layout()
# plt.show()

# # ========== ДОПОЛНИТЕЛЬНЫЙ ГРАФИК: СКОРОСТЬ СХОДИМОСТИ ==========
# plt.figure(figsize=(10, 6))

# # Сглаживание потерь для лучшей визуализации
# window_size = 100
# smooth_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')

# plt.plot(smooth_losses, 'r-', linewidth=2, label='Сглаженные потери')
# plt.plot(losses, 'b-', linewidth=0.5, alpha=0.3, label='Исходные потери')
# plt.title('Сходимость модели при обучении', fontsize=14, fontweight='bold')
# plt.xlabel('Эпоха', fontsize=12)
# plt.ylabel('Потери (MSE)', fontsize=12)
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.yscale('log')
# plt.show()

# # ========== АНАЛИЗ РЕЗУЛЬТАТОВ ==========
# print("\n" + "=" * 50)
# print("АНАЛИЗ РЕЗУЛЬТАТОВ:")
# print("=" * 50)

# # Проверка точности
# predictions = (nn.output > 0.5).astype(int)
# accuracy = np.mean(predictions == y) * 100
# print(f"Точность классификации: {accuracy:.2f}%")

# # Статистика потерь
# print(f"\nНачальные потери: {losses[0]:.6f}")
# print(f"Конечные потери: {losses[-1]:.6f}")
# print(f"Уменьшение потерь: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")

# # Скорость обучения (потери на последних 1000 эпохах)
# recent_losses = losses[-1000:]
# print(f"\nСредние потери на последних 1000 эпохах: {np.mean(recent_losses):.6f}")
# print(f"Стандартное отклонение: {np.std(recent_losses):.6f}")



# Практическое правило
def suggest_neurons(n_samples, n_features, n_classes=None, complexity='medium'):
    """
    Эвристики для выбора количества нейронов
    """
    # Базовое количество
    base = (n_features + (n_classes if n_classes else 10)) // 2
    
    if complexity == 'low':
        return max(4, base)
    elif complexity == 'medium':
        return max(8, base * 2)
    elif complexity == 'high':
        return max(16, base * 4)
    else:
        return base

# Примеры
print("Примеры рекомендаций:")
print(f"MNIST (28x28=784 признака, 10 классов): {suggest_neurons(60000, 784, 10, 'high')} нейронов")
print(f"XOR (2 признака, 2 класса): {suggest_neurons(4, 2, 2, 'low')} нейронов")
print(f"Цены на жилье (13 признаков, регрессия): {suggest_neurons(500, 13, complexity='medium')} нейронов")