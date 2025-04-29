import itertools
val = [1, 2, 3, 4,5,6,7,8,9]
com_set = itertools.combinations(val, 5)
n=0
for elem in com_set:
    print(elem)
    n+=1
C59 = n
print(C59)
val = [1, 2, 3, 4,5,6,7,8,9,10]
com_set = itertools.combinations(val, 6)
n=0
for elem in com_set:
    # print(elem)
    n+=1
C610 = n
print(C610)
P = C59/C610
print(P)

import random

def urn_experiment(num_experiments):
    # Ініціалізуємо кількість білих та чорних куль
    white_balls = 3
    black_balls = 7


    n = white_balls + black_balls


    white_balls_list = ["біла"]*white_balls
    black_balls_list = ["чорна"]*black_balls


    choice_list = white_balls_list + black_balls_list


    # Лічильники подій витягання білої та чорної кулі
    white_count = 0
    black_count = 0


    for _ in range(num_experiments):
        # Витягуємо одну кулю випадковим чином
        selected_ball = random.choice(choice_list)


        # Підрахунок результатів
        if selected_ball == "біла":
            white_count += 1
        else:
            black_count += 1


    # Виведення результатів та відносної частоти
    print("\nСтатистика:")
    print(f"Кількість експериментів: {num_experiments}")
    print(f"Кількість білих куль: {white_count}")
    print(f"Кількість чорних куль: {black_count}")
    print(f"Відносна частота білої кулі: {white_count / (num_experiments):.2f}")
    print(f"Відносна частота чорної кулі: {black_count / (num_experiments):.2f}")


# Введення кількості експериментів
num_experiments = int(input("Введіть кількість експериментів: "))


# Запуск симуляції
urn_experiment(num_experiments)