from math import comb
import matplotlib.pyplot as plt

# Параметри
N = 50      # загальна кількість деталей
K = 5       # нестандартних деталей
n = 6       # вибираємо 6 деталей

# Розрахунок ймовірностей для X = 0,1,2,...,min(K,n)
x_vals = list(range(0, min(K, n) + 1))
probabilities = []

for k in x_vals:
    p = comb(K, k) * comb(N - K, n - k) / comb(N, n)
    probabilities.append(p)

# Вивід ймовірностей у консоль
for x, p in zip(x_vals, probabilities):
    print(f"P(X = {x}) = {p:.4f}")

# Побудова графіку
plt.figure(figsize=(8, 5))
plt.plot(x_vals, probabilities, marker='o', linestyle='-', color='blue')
plt.title("Розподіл ймовірностей (гіпергеометричний)")
plt.xlabel("Кількість нестандартних деталей (X)")
plt.ylabel("Ймовірність P(X)")
plt.grid(True)
plt.xticks(x_vals)
plt.show()

# task 2
from math import comb
import matplotlib.pyplot as plt

# Параметри задачі
n = 7        # кількість пострілів
p = 0.8      # ймовірність влучання
q = 1 - p    # ймовірність промаху

# Обчислення ймовірностей для X = 0...7
x_vals = list(range(n + 1))
probabilities = []

for k in x_vals:
    P = comb(n, k) * (p ** k) * (q ** (n - k))
    probabilities.append(P)

# Ймовірність X ≥ 5
p_ge_5 = sum(probabilities[5:])
print(f"Ймовірність того, що буде не менше 5 влучань: {p_ge_5:.4f}")

# Побудова полігону розподілу
plt.figure(figsize=(8, 5))
plt.plot(x_vals, probabilities, marker='o', linestyle='-', color='green')
plt.title("Полігон розподілу (біноміальний розподіл)")
plt.xlabel("Кількість влучань (X)")
plt.ylabel("Ймовірність P(X)")
plt.grid(True)
plt.xticks(x_vals)
plt.show()

# Task 3

from itertools import product

# Усі можливі комбінації для 3 кубиків (1..6)
all_combinations = list(product(range(1, 7), repeat=3))

# Підрахунок кількості сум
sum_11 = 0
sum_12 = 0

for combo in all_combinations:
    total = sum(combo)
    if total == 11:
        sum_11 += 1
    elif total == 12:
        sum_12 += 1

# Вивід результатів
total_combinations = len(all_combinations)
prob_11 = sum_11 / total_combinations
prob_12 = sum_12 / total_combinations

print(f"Загальна кількість комбінацій: {total_combinations}")
print(f"Кількість комбінацій із сумою 11: {sum_11}, ймовірність: {prob_11:.4f}")
print(f"Кількість комбінацій із сумою 12: {sum_12}, ймовірність: {prob_12:.4f}")

if sum_11 > sum_12:
    print("▶️ Сума 11 імовірніша за суму 12.")
else:
    print("▶️ Сума 12 імовірніша або дорівнює сумі 11.")

# Task 6
import matplotlib.pyplot as plt
import numpy as np

n_trials = 100_000
remaining_white_count = 0
accepted_trials = 0

for _ in range(n_trials):
    initial_ball = np.random.choice(['white', 'black'])
    balls = ['white', initial_ball]
    drawn_ball = np.random.choice(balls)
    
    if drawn_ball == 'white':
        accepted_trials += 1
        balls.remove('white')
        if balls[0] == 'white':
            remaining_white_count += 1

simulated_probability = remaining_white_count / accepted_trials

# Побудова графіка
labels = ['Залишилась біла', 'Залишилась чорна']
probs = [simulated_probability, 1 - simulated_probability]

fig, ax = plt.subplots()
bars = ax.bar(labels, probs, color=['lightgreen', 'lightcoral'])

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords='offset points', ha='center', fontsize=12)

ax.set_ylim(0, 1)
ax.set_ylabel('Ймовірність')
ax.set_title('Симуляція: яка куля залишилася, якщо витягли білу')
plt.grid(axis='y')
plt.show()

# Task 7
import numpy as np
import matplotlib.pyplot as plt

def simulate_walks(n_simulations, steps=3):
    # Одне блукання: на кожному кроці або +2, або -1 з рівною ймовірністю
    outcomes = np.random.choice([2, -1], size=(n_simulations, steps))
    final_prices = outcomes.sum(axis=1)
    return final_prices

# Кількість симуляцій для аналізу
simulations_counts = [10, 100, 1000, 10000]

for n in simulations_counts:
    results = simulate_walks(n)
    mean_price = np.mean(results)
    print(f"Симуляцій: {n}, Середнє значення: {mean_price:.4f}")
    
    plt.hist(results, bins=range(min(results), max(results) + 2), align='left', rwidth=0.8, edgecolor='black')
    plt.title(f'Гістограма цін після 3 кроків ({n} симуляцій)')
    plt.xlabel('Ціна')
    plt.ylabel('Частота')
    plt.grid(True)
    plt.show()
