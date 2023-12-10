from scipy.stats import t, chi2_contingency, chi2, chisquare, norm
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
import math
import random
import matplotlib.pyplot as plt


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
    print(f'|Хмакс - Хсред|/S = {abs(max(content) - mean) / S}, critery = {critery[len(content)]}')
    return abs(max(content) - mean) / S > critery[len(content)]


def check_grubbs_min(content, S, critery, mean):
    print(f'|Хмин - Хсред|/S = {abs(min(content) - mean) / S}, critery = {critery[len(content)]}')
    return abs(min(content) - mean) / S > critery[len(content)]


def offset_square(content, mean):
    res = 0
    for i in content:
        res += (i - mean) ** 2
    res = res / len(content)
    res = res ** 0.5
    print(f'Смещенное среднее квадратическое отклонение = {res}')
    return res


def check_d_critery(content, mean, d_criteries):
    d = 0
    for i in content:
        d += abs(i - mean)
    d = d / (len(content) * offset_square(content, mean))
    print(f'd с "шапкой" = {d}')
    if d_criteries[len(content)][0] < d <= d_criteries[len(content)][1]:
        print(
            f"d(1-1/2) < d <= d(q/2) успешно выполняется: {d_criteries[len(content)][0]} < {d} <= {d_criteries[len(content)][1]}\n")
    return d_criteries[len(content)][0] < d <= d_criteries[len(content)][1]


def check_second_critery(content, S, criteries):
    differences = []
    for i in range(0, len(content) - 1):
        differences.append(abs(content[i] - mean))
        # print(f'Xi = {content[i]}, mean = {mean}, diff = {abs(content[i] - mean)}')
    print(f'z(P/2)= {criteries[len(content)][0]}\nS= {S}')
    check_num = criteries[len(content)][0] * S
    print(f'z(P/2) * S = {check_num}')
    count = 0
    for i in differences:
        if i > check_num:
            count += 1
    print(
        f'допустимое число разниц, превосходящих z(P/2) * S это: {criteries[len(content)][1]}, число разниц, превысивших m: {count}')
    return count <= criteries[len(content)][1]


def normal_density_in_context(x):
    return (1 / (math.sqrt(2 * math.pi))) * math.exp(0.5 * x)


def check_normality_big_nums(content, S, criteries):
    # полагается, что в выборке до сотни элементов
    r_def = 9
    h = (max(content) - min(content)) / r_def
    small_arrays = split_array_by_range(content, h)
    lengths = []
    means = []
    Si = []
    for i in small_arrays:
        lengths.append(len(i))
        means.append(np.mean(i))
        Si.append(np.std(i))
    midpoints = find_midpoints(small_arrays)
    ni = []
    '''for i in small_arrays:
        res = lengths[small_arrays.index(i)] * h * Si[small_arrays.index(i)]
        fi = (midpoints[small_arrays.index(i)] - means[small_arrays.index(i)]) / Si[small_arrays.index(i)]
        dens = normal_density_in_context(fi)
        res = res * fi
        ni.append(res)'''
    for i in small_arrays:
        res = lengths[small_arrays.index(i)] * h * S
        fi = (midpoints[small_arrays.index(i)] - mean) / S
        dens = normal_density_in_context(fi)
        res = res * fi
        ni.append(res)
    xi_square = []
    for i in small_arrays:
        xi = ((lengths[small_arrays.index(i)] - ni[small_arrays.index(i)]) ** 2) / ni[small_arrays.index(i)]
        xi_square.append(xi)
    for i in range(0, len(small_arrays)):
        print(
            f'Интервал: {small_arrays[i]}\nСреднее в интервале: {means[i]}\nДлина интервала: {lengths[i]}\nСерединное значение интервала: {midpoints[i]}\nКоличество предполагаемых точек в нормальном распределении: {ni[i]}\n')
    summary_xi = sum(xi_square)


def find_midpoints(arrays):
    midpoints = []
    for subarray in arrays:
        midpoint_index = len(subarray) // 2
        midpoint_value = subarray[midpoint_index]
        midpoints.append(midpoint_value)
    return midpoints


def find_excess_and_etc(content, S):
    excess = kurtosis(content) + 3
    assimetry = skew(content)
    mean = np.mean(content)
    contr_excess = 1/(excess**0.5)
    return excess, contr_excess, assimetry


def split_array_by_range(content, max_range):
    result = []
    array = content
    array = sorted(array)
    while len(array) != 0:
        array = sorted(array)
        result.append([])
        result[len(result) - 1].append(array[0])
        array.pop(0)
        while max(result[len(result) - 1]) - min(result[len(result) - 1]) <= max_range and len(array) != 0:
            result[len(result) - 1].append(array[0])
            array.pop(0)
        if max(result[len(result) - 1]) - min(result[len(result) - 1]) > max_range and len(
                result[len(result) - 1]) != 1:
            array.append(result[len(result) - 1][-1])
            result[len(result) - 1].pop(-1)

    if len(result[-1]) == 1:
        result[-2].append(result[-1][0])
        result.remove(result[-1])
    return result


# считывание эксель-таблицы с одним коэффициентом
def read_table(filename):
    df = pd.read_excel(filename + '.xlsx')
    # Создание словаря из столбцов A и B
    coefficients_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    return coefficients_dict


# считывание выборки из текстового файла
def read_chosen_nums(filename):
    f = open(filename+'.txt', 'r')
    content = []
    for i in f.readlines():
        if i!='' and i!='\n' and i!=' \n':
            content.append(float(i[:-2].replace(',', '.')))
    f.close()
    return content


def read_two_coeff(filename):
    df = pd.read_excel(filename + '.xlsx', header=None)
    df.iloc[:, 2] = df.iloc[:, 2].replace({',': '.'}, regex=True)
    df.iloc[:, 1] = df.iloc[:, 1].replace({',': '.'}, regex=True)
    coefficients_dict = {}
    for _, row in df.iterrows():
        index_key = int(row.iloc[0])
        coefficients_dict[index_key] = [float(row.iloc[2]), float(row.iloc[1])]
    return coefficients_dict


if __name__ == '__main__':
    content = read_chosen_nums('input')
    grubbs_criteries = read_table('граббс')
    number_of_freedoms = len(content) - 1
    print(f'Размер выборки: {len(content)}')
    mean, S, Sx = avg_square_etc(content)
    while check_grubbs_max(content, S, grubbs_criteries, mean):
        print(
            f'Обнаружен выброс максимального значения ({max(content)}) по критерию Граббса, удаляем значение и пересчитываем параметры выборки')
        content.remove(max(content))
        mean, S, Sx = avg_square_etc(content)
    while check_grubbs_min(content, S, grubbs_criteries, mean):
        print(
            f'Обнаружен выброс минимального значения ({min(content)}) по критерию Граббса, удаляем значение и пересчитываем параметры выборки')
        content.remove(min(content))
        mean, S, Sx = avg_square_etc(content)
    if len(content) <= 50:
        d_criteries = read_two_coeff('квантили критерий 1')
        p_and_z_criteries = read_two_coeff('P для Z')
        if check_d_critery(content, mean, d_criteries) and check_second_critery(content, S, p_and_z_criteries):
            print('Распределение принадлежит нормальному')
        else:
            print('Распределение не принадлежит нормальному')
    else:
        check_normality_big_nums(content, S, mean)
    excess, contr_excess, assimetry = find_excess_and_etc(content, S)
    print(f"Эксцесс: {excess}")
    print(f"Контрэксцесс: {contr_excess}")
    print(f"Ассиметрия: {assimetry}")

    data=content
    plt.hist(data, bins='auto', density=True, alpha=0.7, color='g')
    mu, std = np.mean(data), np.std(data)
    observed_freq, bin_edges = np.histogram(data, bins='auto')
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Центры бинов
    expected_freq = norm.pdf(bin_centers, mu, std) * len(data)
    mu, std = np.mean(data), np.std(data)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = np.exp(-(x - mu) ** 2 / (2 * std ** 2)) / (std * np.sqrt(2 * np.pi))
    plt.plot(x, p, 'k', linewidth=2)

    plt.show()