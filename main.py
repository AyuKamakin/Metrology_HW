import seaborn as sns
from scipy.stats import t, chi2_contingency, chi2, chisquare, norm, shapiro, kstest, anderson, probplot, normaltest, \
    expon, logistic, uniform
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
import math
import matplotlib.pyplot as plt
from scipy.stats import shapiro, anderson, kstest, jarque_bera, cramervonmises
from scipy.stats import linregress, gamma, chi2


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
    print(f'критерий Граббса |Хмакс - Хсред|/S = {abs(max(content) - mean) / S}, критерий Граббса(критическое значение) = {critery[len(content)]}')
    return abs(max(content) - mean) / S > critery[len(content)]


def check_grubbs_min(content, S, critery, mean):
    print(f'критерий Граббса |Хмин - Хсред|/S = {abs(min(content) - mean) / S}, критерий Граббса(критическое значение) = {critery[len(content)]}')
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
def others_check(sample_data):
    alpha=0.05
    # Сравнение с экспоненциальным распределением
    ks_stat_exp, p_value_exp = kstest(sample_data, expon.cdf)
    if p_value_exp > alpha:
        print('Выборка похожа на экспоненциальное распределение по критерию Колмогорова-Смирнова.')
    else:
        print('Выборка не похожа на экспоненциальное распределение по критерию Колмогорова-Смирнова.')

    # Сравнение с равномерным распределением
    ks_stat_uniform, p_value_uniform = kstest(sample_data, uniform.cdf)
    if p_value_uniform > alpha:
        print('Выборка похожа на равномерное распределение по критерию Колмогорова-Смирнова.')
    else:
        print('Выборка не похожа на равномерное распределение по критерию Колмогорова-Смирнова.')

    # Сравнение с логистическим распределением
    ks_stat_logistic, p_value_logistic = kstest(sample_data, logistic.cdf)
    if p_value_logistic > alpha:
        print('Выборка похожа на логистическое распределение по критерию Колмогорова-Смирнова.')
    else:
        print('Выборка не похожа на логистическое распределение по критерию Колмогорова-Смирнова.')
    a, loc, scale = gamma.fit(sample_data)
    ks_stat_gamma, p_value_gamma = kstest(sample_data, gamma(a, loc=loc, scale=scale).cdf)
    if p_value_gamma > alpha:
        print('Выборка похожа на гамма-распределение по критерию Колмогорова-Смирнова.')
    else:
        print('Выборка не похожа на гамма-распределение по критерию Колмогорова-Смирнова.')

    # Сравнение с распределением хи-квадрат
    df_chi2, loc_chi2, scale_chi2 = chi2.fit(sample_data)
    ks_stat_chi2, p_value_chi2 = kstest(sample_data, chi2(df_chi2, loc=loc_chi2, scale=scale_chi2).cdf)
    if p_value_chi2 > alpha:
        print('Выборка похожа на распределение хи-квадрат по критерию Колмогорова-Смирнова.')
    else:
        print('Выборка не похожа на распределение хи-квадрат по критерию Колмогорова-Смирнова.')

    # Сравнение с распределением Стьюдента
    df_t, loc_t, scale_t = t.fit(sample_data)
    ks_stat_t, p_value_t = kstest(sample_data, t(df_t, loc=loc_t, scale=scale_t).cdf)
    if p_value_t > alpha:
        print('Выборка похожа на распределение Стьюдента по критерию Колмогорова-Смирнова.')
    else:
        print('Выборка не похожа на распределение Стьюдента по критерию Колмогорова-Смирнова.')

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    probplot(sample_data, dist=expon,sparams=(len(content)-1,), plot=plt)
    plt.title('Сравнение с экспоненциальным распределением')

    # График сравнения с распределением Стьюдента
    plt.subplot(2, 3, 2)
    probplot(sample_data, dist=t, sparams=(len(content)-1,), plot=plt)
    plt.title('Сравнение с распределением Стьюдента')

    # График сравнения с хи-квадрат распределением
    plt.subplot(2, 3, 4)
    probplot(sample_data, dist=chi2, sparams=(len(content)-1,), plot=plt)
    plt.title('Сравнение с хи-квадрат распределением')

    # График сравнения с равномерным распределением
    plt.subplot(2, 3, 3)
    probplot(sample_data, dist=uniform, sparams=(len(content)-1,), plot=plt)
    plt.title('Сравнение с равномерным распределением')

    # График сравнения с равномерным распределением
    plt.subplot(2, 3, 5)
    probplot(sample_data, dist=gamma, sparams=(len(content)-1,), plot=plt)
    plt.title('Сравнение с гамма распределением')

    # График сравнения с равномерным распределением
    plt.subplot(2, 3, 6)
    probplot(sample_data, dist=logistic, sparams=(len(content) - 1,), plot=plt)
    plt.title('Сравнение с логистическим распределением')

    plt.tight_layout()
    plt.show()
def normality_check(data):
    statistic_shapiro, p_value_shapiro = shapiro(data)
    alpha = 0.05
    # Проверка нормальности с помощью критерия Андерсона-Дарлинга
    result_anderson = anderson(data)
    # Проверка нормальности с помощью критерия Колмогорова-Смирнова
    statistic_ks, p_value_ks = kstest(data, 'norm')
    # Проверка нормальности с помощью критерия Крамера-фон-Мизеса
    result_cvm = cramervonmises(data, cdf='norm')
    statistic_cvm, p_value_cvm = result_cvm.statistic, result_cvm.pvalue
    # Тест на нормальность основанный на хи-квадрат Пирсона
    statistic_test, p_value_test = normaltest(data)
    # Критерий Харке-Бера
    statistic_jb, p_value_jb = jarque_bera(data)

   # print(f'Статистика критерия Харке-Бера: {statistic_jb:.4f}, p-value: {p_value_jb:.4f}\n \n')
    #  print(f"Статистика теста, основанного на критерии хи-квадрат Пирсона: {statistic_test:.4f}, p-value: {p_value_test:.4f}")
    #  print(f'Статистика критерия Крамера-фон-Мизеса:  {statistic_cvm:.4f}, p-value: {p_value_cvm:.4f}')
    #  print(f'Статистика критерия Колмогорова-Смирнова: {statistic_ks:.4f}, p-value: {p_value_ks:.4f}')
    # print(f'Статистика критерия Андерсона-Дарлинга: {result_anderson.statistic:.4f}, Critical Values: {result_anderson.critical_values}')
    #  print(f'Статистика критерия Шапиро-Вилка: {statistic_shapiro:.4f}, p-value: {p_value_shapiro:.4f}')

    count=0
    if p_value_shapiro > alpha:
        print('Выборка согласуется с нормальным распределением по критерию Шапиро-Вилка,\nотклонений в центральной части не наблюдается')
        count+=1
    else:
        print('Выборка не согласуется с нормальным распределением по критерию Шапиро-Вилка,\nнаблюдается отклонение в центральной части')
    if p_value_ks > alpha:
        print('Выборка согласуется с нормальным распределением по критерию Колмогорова-Смирнова.')
        count+=1
    else:
        print('Выборка не согласуется с нормальным распределением по критерию Колмогорова-Смирнова.')
    if result_anderson.statistic < result_anderson.critical_values[2]:
        print('Выборка согласуется с нормальным распределением по критерию Андерсона-Дарлинга.')
        count+=1
    else:
        print('Выборка не согласуется с нормальным распределением по критерию Андерсона-Дарлинга.')
    if p_value_cvm > alpha:
        print('Выборка согласуется с нормальным распределением по критерию Крамера-фон-Мизеса.')
        count+=1
    else:
        print('Выборка не согласуется с нормальным распределением по критерию Крамера-фон-Мизеса.')
    if p_value_jb > alpha:
        print('Выборка согласуется с нормальным распределением по критерию Харке-Бера.')
        count+=1
    else:
        print('Выборка не согласуется с нормальным распределением по критерию Харке-Бера.')
    if p_value_test > alpha:
        print("Выборка согласуется с нормальным распределением по тесту, основанному на критерии Хи-квадрат Пирсона.")
        count+=1
    else:
        print(
            "Выборка не согласуется с нормальным распределением по тесту, основанному на критерии Хи-квадрат Пирсона.")
    print(f'Выборка согласуется с нормальным распределением по {count} из 6 критериев. \n \n')

    data=content
    plt.figure(figsize=(12, 6))
    # Гистограмма
    plt.subplot(1, 2, 1)
    plt.hist(data, bins='auto', density=True, alpha=0.7, color='g')
    mu, std = np.mean(data), np.std(data)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = np.exp(-(x - mu) ** 2 / (2 * std ** 2)) / (std * np.sqrt(2 * np.pi))
    plt.plot(x, p, 'k', linewidth=2)
    plt.title('Гистограмма и нормальное распределение')

    # Q-Q график
    plt.subplot(1, 2, 2)
    probplot(data, dist='norm', plot=plt)
    plt.title('Q-Q график')

    # Отображение графиков
    plt.tight_layout()
    plt.show()
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
    content = read_chosen_nums('input2')
    grubbs_criteries = read_table('граббс')
    number_of_freedoms = len(content) - 1
    print(f'Размер выборки: {len(content)}')
    mean, S, Sx = avg_square_etc(content)
    old_len=len(content)
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
    if old_len==len(content):
        print("Выбросов по критерию Граббса не обнаружено")
    '''if len(content) <= 50:
        d_criteries = read_two_coeff('квантили критерий 1')
        p_and_z_criteries = read_two_coeff('P для Z')
        if check_d_critery(content, mean, d_criteries) and check_second_critery(content, S, p_and_z_criteries):
            print('Распределение принадлежит нормальному')
        else:
            print('Распределение не принадлежит нормальному')
    else:
        check_normality_big_nums(content, S, mean)'''
    excess, contr_excess, assimetry = find_excess_and_etc(content, S)
    print(f"Эксцесс: {excess}")
    print(f"Контрэксцесс: {contr_excess}")
    print(f"Ассиметрия: {assimetry}")
    normality_check(content)
    others_check(content)