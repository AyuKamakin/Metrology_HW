import numpy as np
from scipy.stats import t
import pandas as pd


def avg_square_etc(content):
    n = len(content)
    # среднее
    mean = sum(content) / n
    average_square_distortion = 0
    for j in content:
        average_square_distortion += (j - mean) ** 2
    # СКО
    average_square_distortion = (average_square_distortion / (n - 1)) ** 0.5
    # СКО среднего арифметического
    average_square_distortion_of_mean = average_square_distortion / (n ** 0.5)
    print(
        f"Оценка измеряемой величины: {mean} \nСКО выборки {average_square_distortion} \nСко среднего арифметического {average_square_distortion_of_mean}")
    return mean, average_square_distortion, average_square_distortion_of_mean

def check_grubbs_max(content, S, critery, mean):
    print(f'abs(Max-mean)/S = {abs(max(content) - mean) / S}, critery = {critery[len(content)]}')
    return abs(max(content) - mean) / S > critery[len(content)]


def check_grubbs_min(content, S, critery, mean):
    print(f'abs(Max-mean)/S = {abs(min(content) - mean) / S}, critery = {critery[len(content)]}')
    return abs(min(content) - mean) / S > critery[len(content)]

def offset_square(content, mean):
    res = 0
    for i in content:
        res += (i - mean) ** 2
    res = res/len(content)
    res = res**0.5
    print(f'Смещенное среднее квадратическое отклонение = {res}')
    return res

def check_d_critery(content, mean, d_criteries):
    d=0
    for i in content:
        d+=abs(i-mean)
    d=d/(len(content)*offset_square(content,mean))
    print(f'd с "шапкой" = {d}')
    if d_criteries[len(content)][0] < d <= d_criteries[len(content)][1]:
        print(f"d(1-1/2) < d <= d(q/2) успешно выполняется: {d_criteries[len(content)][0]} < {d} <= {d_criteries[len(content)][1]}\n")
    return d_criteries[len(content)][0] < d <= d_criteries[len(content)][1]

def check_second_critery(content, S, criteries):
    differences=[]
    for i in range(0, len(content)-1):
        differences.append(abs(content[i]-mean))
    count = 0
    print(f'z(P/2)= {...}')
    #check_num= z* S
    for i in differences:
        if i>check_num:
            count+=1
    #m=...
    print(f'допустимое число разниц, превосходящих z(P/2)*S это: {m}, число разниц, превысивших m: {count}')
    return count<=m

# считывание эксель-таблицы с коэффициентами Граббса
def read_grubbs_table():
    df = pd.read_excel('граббс.xlsx')
    # Создание словаря из столбцов A и B
    coefficients_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    return coefficients_dict


# считывание выборки из текстового файла
def read_chosen_nums():
    f = open('input.txt', 'r')
    content = []
    for i in f.readlines():
        content.append(float(i[:-2].replace(',', '.')))
    f.close()
    return content

def read_d_critery():
    df = pd.read_excel('квантили критерий 1.xlsx', header=None)
    df.iloc[:, 2] = df.iloc[:, 2].replace({',': '.'}, regex=True)
    df.iloc[:, 1] = df.iloc[:, 1].replace({',': '.'}, regex=True)
    # Создаем словарь
    coefficients_dict = {}
    for _, row in df.iterrows():
        index_key = int(row.iloc[0])
        coefficients_dict[index_key] = [float(row.iloc[2]), float(row.iloc[1])]
    return coefficients_dict
if __name__ == '__main__':
    content = read_chosen_nums()
    grubbs_criteries = read_grubbs_table()
    number_of_freedoms = len(content) - 1
    print(f'Размер выборки: {len(content)}')
    mean, S, Sx = avg_square_etc(content)
    check_1 = check_grubbs_max(content, S, grubbs_criteries, mean)
    while (check_1):
        print(
            f'Обнаружен выброс максимального значения ({max(content)}) по критерию Граббса, удаляем значение и пересчитываем параметры выборки')
        content.remove(max(content))
        mean, S, Sx = avg_square_etc(content)

    check_2 = check_grubbs_min(content, S, grubbs_criteries, mean)
    while (check_2):
        print(
            f'Обнаружен выброс минимального значения ({min(content)}) по критерию Граббса, удаляем значение и пересчитываем параметры выборки')
        content.remove(min(content))
        mean, S, Sx = avg_square_etc(content)
    if len(content) <= 50:
        d_criteries=read_d_critery()
        if check_d_critery(content, mean, d_criteries) and check_second_critery(content, S, ):
            print('Распределение принадлежит нормальному')
        else:
            print('Распределение не принадлежит нормальному')

