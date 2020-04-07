import numpy as np
import math
from scipy.stats import t, f

N = 8
m = 3
q = 0.05

min_x1, max_x1, min_x2, max_x2, min_x3, max_x3 = -30, 0, -35, 10, 0, 20
mean_Xmax = (max_x1 + max_x2 + max_x3) / 3
mean_Xmin = (min_x1 + min_x2 + min_x3) / 3
max_y = round(200 + mean_Xmax)
min_y = round(200 + mean_Xmin)

flag = False


def combine(x0, x1, x2, x3):
    x1x2 = [a * b for a, b in zip(x1, x2)]
    x1x3 = [a * b for a, b in zip(x1, x3)]
    x2x3 = [a * b for a, b in zip(x2, x3)]
    x1x2x3 = [a * b * c for a, b, c in zip(x1, x2, x3)]
    return np.array([x0, x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3]).T


def mean_y(N, m, matrix):
    sumY_matrix, meanY_matrix = np.zeros(N), np.zeros(N)
    for i in range(N):
        for j in range(m):
            sumY_matrix[i] += matrix[i][j]
            meanY_matrix[i] = round(sumY_matrix[i] / m, 3)
    return meanY_matrix


def dispersion(m, N, y, meanY):
    sigma = np.zeros(N)
    for i in range(N):
        for j in range(m):
            sigma[i] += pow((y[i][j] - meanY[i]), 2)
        sigma[i] = sigma[i] / m
    return sigma


def cochren(m, N, q, sigma):
    f1, f2 = m - 1, N
    f_value = f.ppf(1 - q / f1, f2, (f1 - 1) * f2)
    Gt = f_value / (f_value + f1 - 1)
    Gp = max(sigma) / sum(sigma)
    print("Критерій Кохрена:")
    if Gp < Gt:
        print("{} < {} => Дисперсія однорідна.".format(round(Gp, 3), round(Gt, 3)))
    else:
        print("{} > {}Дисперсія неоднорідна.".format(round(Gp, 3), round(Gt, 3)))
        return False


def equation(min_x1, max_x1, min_x2, max_x2, min_x3, max_x3, y_matrix, N, m, q):
    # нормовані значення факторів (з фіктивним нульовим фактором x0=1)
    x_coded = np.array([[1, -1, -1, -1],
                        [1, -1, 1, 1],
                        [1, 1, -1, 1],
                        [1, 1, 1, -1]])

    # Значення ф-ції відгуку при N=4
    y_matrix = y_matrix[:4]

    # натуралізовані значення факторів
    x_matrix = np.array([[min_x1, min_x2, min_x3],
                         [min_x1, max_x2, max_x3],
                         [max_x1, min_x2, max_x3],
                         [max_x1, max_x2, min_x3]])

    meanY_matrix = mean_y(N, m, y_matrix)

    mx1, mx2, mx3 = np.sum(x_matrix, axis=0)[0] / N, np.sum(x_matrix, axis=0)[1] / N, np.sum(x_matrix, axis=0)[2] / N

    my = np.sum(meanY_matrix) / N

    a1, a2, a3, a11, a22, a33, a12, a13, a23 = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(len(x_matrix)):
        a1 += x_matrix[i][0] * meanY_matrix[i] / len(x_matrix)
        a2 += x_matrix[i][1] * meanY_matrix[i] / len(x_matrix)
        a3 += x_matrix[i][2] * meanY_matrix[i] / len(x_matrix)

        a11 += x_matrix[i][0] ** 2 / len(x_matrix)
        a22 += x_matrix[i][1] ** 2 / len(x_matrix)
        a33 += x_matrix[i][2] ** 2 / len(x_matrix)

        a12 += x_matrix[i][0] * x_matrix[i][1] / len(x_matrix)
        a13 += x_matrix[i][0] * x_matrix[i][2] / len(x_matrix)
        a23 += x_matrix[i][1] * x_matrix[i][2] / len(x_matrix)

    a21 = a12
    a31 = a13
    a32 = a23

    determ = np.linalg.det([[1, mx1, mx2, mx3],
                            [mx1, a11, a12, a13],
                            [mx2, a12, a22, a32],
                            [mx3, a13, a23, a33]])

    determ0 = np.linalg.det([[my, mx1, mx2, mx3],
                             [a1, a11, a12, a13],
                             [a2, a12, a22, a32],
                             [a3, a13, a23, a33]])

    determ1 = np.linalg.det([[1, my, mx2, mx3],
                             [mx1, a1, a12, a13],
                             [mx2, a2, a22, a32],
                             [mx3, a3, a23, a33]])

    determ2 = np.linalg.det([[1, mx1, my, mx3],
                             [mx1, a11, a1, a13],
                             [mx2, a12, a2, a32],
                             [mx3, a13, a3, a33]])

    determ3 = np.linalg.det([[1, mx1, mx2, my],
                             [mx1, a11, a12, a1],
                             [mx2, a12, a22, a2],
                             [mx3, a13, a23, a3]])

    # коефіцієнти рівняння регресії
    b0, b1, b2, b3 = determ0 / determ, determ1 / determ, determ2 / determ, determ3 / determ
    print("Рівняння регресії: y = {} + {}*x1 + {}*x2 + {}*x3".format(round(b0, 3), round(b1, 3), round(b2, 3),
                                                                         round(b3, 3)))
    # Перевірка
    y1 = b0 + b1 * x_matrix[0][0] + b2 * x_matrix[0][1] + b3 * x_matrix[0][2]
    y2 = b0 + b1 * x_matrix[1][0] + b2 * x_matrix[1][1] + b3 * x_matrix[1][2]
    y3 = b0 + b1 * x_matrix[2][0] + b2 * x_matrix[2][1] + b3 * x_matrix[2][2]
    y4 = b0 + b1 * x_matrix[3][0] + b2 * x_matrix[3][1] + b3 * x_matrix[3][2]

    # Критерій Кохрена (перша статистична перевірка)
    # Дисперсія
    sigma = dispersion(m, N, y_matrix, meanY_matrix)
    if cochren(m, N, q, sigma) == False:
        return False

    # Критерій Стьюдента (Друга статистична перевірка)
    f1, f2 = m - 1, N
    f3 = f1 * f2
    # Оцінка генеральної дисперсії відтворюваності
    Sb = sum(sigma) / N
    Sbs_2 = Sb / (N * m)
    Sbs = math.sqrt(Sbs_2)

    # Оцінки коефіцієнтів
    beta = np.zeros(N)
    for j in range(N):
        beta[j] = sum([x_coded[i][j] * meanY_matrix[i] for i in range(len(x_coded))]) / N
    T0, T1, T2, T3 = abs(beta[0]) / Sbs, abs(beta[1]) / Sbs, abs(beta[2]) / Sbs, abs(beta[3]) / Sbs

    Tt = t.ppf((1 + (1 - q)) / 2, f3)
    # Перевірка значущості коефіцієнтів (b0, b1, b2, b3) рівняння регресії (Якщо відповідний T > Tt => коеф. значущий, інакше він "=" 0
    b_0 = b0 if T0 > Tt else 0
    b_1 = b1 if T1 > Tt else 0
    b_2 = b2 if T2 > Tt else 0
    b_3 = b3 if T3 > Tt else 0
    beta_matrix = np.array([b_0, b_1, b_2, b_3])

    #Перевірка
    y_1 = b_0 + b_1 * x_matrix[0][0] + b_2 * x_matrix[0][1] + b_3 * x_matrix[0][2]
    y_2 = b_0 + b_1 * x_matrix[1][0] + b_2 * x_matrix[1][1] + b_3 * x_matrix[1][2]
    y_3 = b_0 + b_1 * x_matrix[2][0] + b_2 * x_matrix[2][1] + b_3 * x_matrix[2][2]
    y_4 = b_0 + b_1 * x_matrix[3][0] + b_2 * x_matrix[3][1] + b_3 * x_matrix[3][2]
    y_list = [y_1, y_2, y_3, y_4]

    # Критерій Фішера (Третя статистична перевірка)
    d = len(beta_matrix[np.array(beta_matrix) != 0])
    f4 = N - d
    # Дисперсія адекватності
    Sad = m / (N - d) * sum([(y_list[i] - meanY_matrix[i]) ** 2 for i in range(len(meanY_matrix))])
    Fp = Sad / Sbs_2
    Ft = f.ppf(1 - q, f4, f3)
    if Fp > Ft:
        print("Рівняння регресії неадекватне оригіналу при q = {}\n{} > {}".format(round(q, 3), round(Fp, 3),
                                                                                   round(Ft, 3)))
        return False
    else:
        print("Рівняння регресії адекватне оригіналу при q = {}\n{} < {}".format(round(q, 3), round(Fp, 3), round(Ft, 3)))
        return True


def interaction(min_x1, max_x1, min_x2, max_x2, min_x3, max_x3, N, m, q):
    mean_Xmax, mean_Xmin = (max_x1 + max_x2 + max_x3) / 3, (min_x1 + min_x2 + min_x3) / 3
    max_y, min_y = round(200 + mean_Xmax), round(200 + mean_Xmin)
    # генеруємо у
    y_matrix = np.random.randint(min_y, max_y, size=(N, m))
    print("Матриця ігреків\n", y_matrix)
    linear_equation = equation(min_x1, max_x1, min_x2, max_x2, min_x3, max_x3, y_matrix, N - 4, m, q)
    if not linear_equation:
        # нормовані значення факторів (з фіктивним нульовим фактором x0=1)
        x0_coded = [1, 1, 1, 1, 1, 1, 1, 1]
        x1_coded = [-1, -1, 1, 1, -1, -1, 1, 1]
        x2_coded = [-1, 1, -1, 1, -1, 1, -1, 1]
        x3_coded = [-1, 1, 1, -1, 1, -1, -1, 1]
        # генеруємо комбінації
        x_coded = combine(x0_coded, x1_coded, x2_coded, x3_coded)
        # середні значення у
        meanY_matrix = mean_y(N, m, y_matrix)

        print("Нормалізовані фактори (з урахуванням комбінацій)", x_coded)
        b1 = []
        for j in range(N):
            s = 0
            for i in range(N):
                s += (x_coded[i][j] * meanY_matrix[i]) / N
            b1.append(s)
        print("Нормоване рівняння регресії з ефектом взаємодії:")
        print("y = {} + {} * x1 + {} * x2 +{} * x3 + {} * x1x2 + {} * x1x3 + {} * x2x3 + {} * x1x2x3".format(
            round(b1[0], 3), round(b1[1], 3), round(b1[2], 3), round(b1[3], 3), round(b1[4], 3), round(b1[5], 3),
            round(b1[6], 3), round(b1[7], 3)))

        # натуралізовані значення факторів
        x0 = [1, 1, 1, 1, 1, 1, 1, 1]
        x1 = [min_x1, min_x1, max_x1, max_x1, min_x1, min_x1, max_x1, max_x1]
        x2 = [min_x2, max_x2, min_x2, max_x2, min_x2, max_x2, min_x2, max_x2]
        x3 = [min_x3, max_x3, max_x3, min_x3, max_x3, min_x3, min_x3, max_x3]
        # генеруємо комбінації
        x_naturalized = combine(x0, x1, x2, x3)

        print("Натуралізовані фактори (з урахуванням комбінацій)", x_naturalized)

        b2 = np.linalg.solve(x_naturalized, meanY_matrix)
        print("Натуралізоване рівняння регресії з ефектом взаємодії:")
        print("y = {} + {} * x1 + {} * x2 +{} * x3 + {} * x1x2 + {} * x1x3 + {} * x2x3 + {} * x1x2x3".format(
            round(b2[0], 3), round(b2[1], 3), round(b2[2], 3), round(b2[3], 3), round(b2[4], 3), round(b2[5], 3),
            round(b2[6], 3), round(b2[7], 3)))

        # Критерій Кохрена
        # Дисперсія
        sigma = dispersion(m, N, y_matrix, meanY_matrix)
        if cochren(m, N, q, sigma) == False:
            return False

        # Критерій Стьюдента (Друга статистична перевірка)
        f1, f2 = m - 1, N
        f3 = f1 * f2

        # Оцінка генеральної дисперсії відтворюваності
        Sb = sum(sigma) / N
        Sbs_2 = Sb / (N * m)
        Sbs = math.sqrt(Sbs_2)

        # оцінки коефіцієнтів
        beta = np.zeros(N)
        for j in range(N):
            beta[j] = sum([x_coded[i][j] * meanY_matrix[i] for i in range(len(x_coded))]) / N

        Tt = t.ppf((1 + (1 - q)) / 2, f3)
        # Перевірка значущості коефіцієнтів (b0..b7) рівняння регресії
        # (Якщо відповідний T > Tt => коеф. значущий, інакше він "=" 0
        T = np.zeros(N)
        b_ = np.zeros(N)
        for i in range(N):
            T[i] = abs(beta[i]) / Sbs
            b_[i] = beta[i] if T[i] > Tt else 0

        # Перевірка
        y_0 = b_[0] + b_[1] * x_naturalized[0][1] + b_[2] * x_naturalized[0][2] + b_[3] * x_naturalized[0][3] + \
                  b_[4] * x_naturalized[0][4] + b_[5] * x_naturalized[0][5] + b_[6] * x_naturalized[0][6] + \
                  b_[7] * x_naturalized[0][7]
        y_1 = b_[0] + b_[1] * x_naturalized[1][1] + b_[2] * x_naturalized[1][2] + b_[3] * x_naturalized[1][3] + \
              b_[4] * x_naturalized[1][4] + b_[5] * x_naturalized[1][5] + b_[6] * x_naturalized[1][6] + \
              b_[7] * x_naturalized[1][7]
        y_2 = b_[0] + b_[1] * x_naturalized[2][1] + b_[2] * x_naturalized[2][2] + b_[3] * x_naturalized[2][3] + \
              b_[4] * x_naturalized[2][4] + b_[5] * x_naturalized[2][5] + b_[6] * x_naturalized[2][6] + \
              b_[7] * x_naturalized[2][7]
        y_3 = b_[0] + b_[1] * x_naturalized[3][1] + b_[2] * x_naturalized[3][2] + b_[3] * x_naturalized[3][3] + \
              b_[4] * x_naturalized[3][4] + b_[5] * x_naturalized[3][5] + b_[6] * x_naturalized[3][6] + \
              b_[7] * x_naturalized[3][7]
        y_4 = b_[0] + b_[1] * x_naturalized[4][1] + b_[2] * x_naturalized[4][2] + b_[3] * x_naturalized[4][3] + \
              b_[4] * x_naturalized[4][4] + b_[5] * x_naturalized[4][5] + b_[6] * x_naturalized[4][6] + \
              b_[7] * x_naturalized[4][7]
        y_5 = b_[0] + b_[1] * x_naturalized[5][1] + b_[2] * x_naturalized[5][2] + b_[3] * x_naturalized[5][3] + \
              b_[4] * x_naturalized[5][4] + b_[5] * x_naturalized[5][5] + b_[6] * x_naturalized[5][6] + \
              b_[7] * x_naturalized[5][7]
        y_6 = b_[0] + b_[1] * x_naturalized[6][1] + b_[2] * x_naturalized[6][2] + b_[3] * x_naturalized[6][3] + \
              b_[4] * x_naturalized[6][4] + b_[5] * x_naturalized[6][5] + b_[6] * x_naturalized[6][6] + \
              b_[7]* x_naturalized[6][7]
        y_7 = b_[0] + b_[1] * x_naturalized[7][1] + b_[2] * x_naturalized[7][2] + b_[3] * x_naturalized[7][3] + \
              b_[4] * x_naturalized[7][4] + b_[5] * x_naturalized[7][5] + b_[6] * x_naturalized[7][6] + \
              b_[7] * x_naturalized[7][7]
        y_list = [y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7]

        # Критерій Фішера (Третя статистична перевірка)
        d = len(b_[np.array(b_) != 0])
        f4 = N - d
        # Дисперсія адекватності
        Sad = m / (N - d) * sum([(y_list[i] - meanY_matrix[i]) ** 2 for i in range(len(meanY_matrix))])
        Fp = Sad / Sbs_2
        Ft = f.ppf(1 - q, f4, f3)

        if Fp > Ft:
            print("Рівняння регресії неадекватне оригіналу при q = {}\n{} > {}".format(round(q, 3), round(Fp, 3),
                                                                                   round(Ft, 3)))
            flag2 = False
            while not flag2:
                print("#"*120)
                flag2 = interaction(min_x1, max_x1, min_x2, max_x2, min_x3, max_x3, N, m, q)
                if not flag2:
                    m += 1
        else:
            print("Рівняння регресії адекватне оригіналу при q = {}\n{} < {}".format(round(q, 3), round(Fp, 3),
                                                                                     round(Ft, 3)))
        return True
    else:
        return True


while not flag:
    flag = interaction(min_x1, max_x1, min_x2, max_x2, min_x3, max_x3, N, m, q)
    if not flag:
        m += 1
