
import pandas as pd
import numpy as np
import math

excel_data = pd.read_excel('.\lab2.xlsx')
data = pd.DataFrame(excel_data).to_numpy()
print("Матрица данных размером 58x10:\n", pd.DataFrame(data))

n = 58
p = 10


z = [i for i in range(0, p)]    #среднее значение j-го признак
for j in range(0, p):
    for i in range(0, n):
        z[j] += data[i][j]
    z[j] /= n
print("Среднее значение j-го признака\n", pd.DataFrame(z))


disp = [i for i in range(0, p)]     #дисперсия по столбцам
for j in range(0, p):
    for i in range(0, n):
        disp[j] += (data[i][j] - z[j])**2
    disp[j] = disp[j] / n
print("Дисперсия по столбцам:\n", pd.DataFrame(disp))


cov_mtrx = [[0] * p for i in range(p)]  # ковариационная матрица
for i in range(0, p):
    for j in range(0, p):
        for k in range(0, n):
            cov_mtrx[i][j] += (data[k][i] - z[i]) * (data[k][j] - z[j])
        cov_mtrx[i][j] /= n
print("Ковариационная матрица\n", pd.DataFrame(cov_mtrx))


stnd_mtrx = [[0] * p for i in range(n)]  # стандартизованная матрица
for i in range(0, n):
    for j in range(0, p):
        stnd_mtrx[i][j] = (data[i][j] - z[j]) / (disp[j]**0.5)
print("Стандартизованная матрица\n", pd.DataFrame(stnd_mtrx))



corr_mtrx = [[0] * p for i in range(p)]         #корреляционная матрица
for i in range(0, p):
    for j in range(0, p):
        for k in range(0, n):
            corr_mtrx[i][j] += stnd_mtrx[k][i] * stnd_mtrx[k][j]
        corr_mtrx[i][j] /= n
print("Корреляционная матрица\n", pd.DataFrame(corr_mtrx))



alfa = 0.05    #приемлемая вероятность ошибки
t_tab = 1.9954689    #значение t-критерия Стьюдента при уровне значимости 0.05

corr_estimation = [[0] * p for i in range(p)]  # оценка значимости коэффициентов корреляции
for i in range(0, p):
    for j in range(0, p):
        
        if i == j:
            corr_estimation[i][j] = "*"
        else:
            t_calc = (corr_mtrx[i][j] * (n - 2)**0.5) / ((1 - corr_mtrx[i][j] * corr_mtrx[i][j])**0.5)
            if (abs(t_calc) >= t_tab):
                corr_estimation[i][j] = "H1"  # связь между признаками есть
            else:
                corr_estimation[i][j] = "H0"  # связи между признаками нет
print("Оценка значимости коэффициентов корреляции\n", pd.DataFrame(corr_estimation))
