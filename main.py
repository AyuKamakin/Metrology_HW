import random
from scipy.stats import (
    norm,
    probplot,
    uniform,
    triang,
    laplace,
)
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
from scipy.stats import kstest
from scipy.stats import gamma
import statsmodels.api as sm
from tabulate import tabulate


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
        f"Оценка измеряемой величины: {mean} \nСКО выборки {average_square_distortion} \nСко среднего арифметического {average_square_distortion_of_mean}"
    )
    return mean, average_square_distortion, average_square_distortion_of_mean


def check_grubbs_max(content, S, critery, mean):
    print(
        f"критерий Граббса |Хмакс - Хсред|/S = {abs(max(content) - mean) / S}, критерий Граббса(критическое значение) = {critery[len(content)]}"
    )
    return abs(max(content) - mean) / S > critery[len(content)]


def check_grubbs_min(content, S, critery, mean):
    print(
        f"критерий Граббса |Хмин - Хсред|/S = {abs(min(content) - mean) / S}, критерий Граббса(критическое значение) = {critery[len(content)]}"
    )
    return abs(min(content) - mean) / S > critery[len(content)]


def offset_square(content, mean):
    res = 0
    for i in content:
        res += (i - mean) ** 2
    res = float(res) / float(len(content))
    res = res ** 0.5
    print(f"Смещенное среднее квадратическое отклонение = {res}")
    return res


def check_d_critery(content, mean, d_criteries):
    d = 0
    for i in content:
        d += abs(i - mean)
    d = float(d) / float((len(content) * offset_square(content, mean)))
    print(f'd с "шапкой" = {d}')
    if d_criteries[len(content)][0] < d <= d_criteries[len(content)][1]:
        print(
            f"d(1-1/2) < d <= d(q/2) успешно выполняется: {d_criteries[len(content)][0]} < {d} <= {d_criteries[len(content)][1]}\n"
        )
    return (d_criteries[len(content)][0] < d <= d_criteries[len(content)][1])


def check_second_critery(content, S, p_criteries, z_criteries):
    differences = []
    for i in range(0, len(content) - 1):
        differences.append(abs(content[i] - mean))
    print(f"z(P/2)= {p_criteries[len(content)][0]}\nS= {S}")
    check_num = z_criteries[p_criteries[len(content)][0]] * S
    print(f"z(P/2) * S = {check_num}")
    count = 0
    for i in differences:
        if abs(i) > check_num:
            count += 1
    print(
        f"допустимое число разниц, превосходящих z(P/2) * S это: {p_criteries[len(content)][1]}, число разниц, превысивших m: {count}"
    )
    return count <= p_criteries[len(content)][1]

def check_normality_big_nums(content, S, mean, criteries):
    # полагается, что в выборке до тысячи элементов
    if len(content) <= 100:
        r_def = 9
    elif 100 < len(content) <= 1000:
        r_def = random.randint(8, 12)
    f = r_def - 3
    h = (max(content) - min(content)) / r_def
    small_arrays = split_array_by_range(content, h)
    midpoints = find_midpoints(small_arrays)
    table_content = []
    for i in small_arrays:
        xminx = midpoints[small_arrays.index(i)] - mean
        phi_arg = xminx / S
        ni = norm.pdf(phi_arg, loc=0, scale=1) * len(content) * h / S
        xi = ((len(i) - ni) ** 2) / ni
        table_content.append(
            [
                small_arrays.index(i),
                midpoints[small_arrays.index(i)],
                len(i),
                xminx,
                phi_arg,
                ni,
                xi,
            ]
        )
    sum_xi = 0
    for i in table_content:
        sum_xi += i[-1]
    headers = [
        "Номер интервала i",
        "Середина интервала Xio",
        "Число результатов измерений в интервале",
        "Xio-Xср",
        "Zi=Xio-Xср/S",
        "ni",
        "xi2",
    ]
    print_table(headers, table_content)
    print(f"Суммарный xi2 выборки равен: {sum_xi}")
    keys = list(criteries.keys())
    k = 0
    while not (keys[k] < f <= keys[k + 1]):
        k += 1
    print(
        f"Допустимое значение суммарного хи-квадрат для {r_def} интервалов, т.е. {f} степеней свободы составляет {criteries[keys[k+1]]}\nСуммарное значение хи-квадрат: {sum_xi}"
    )
    if sum_xi<=criteries[keys[k+1]]:
        print(
            f'Сумма хи-квадрат {sum_xi} <= {criteries[keys[k+1]]} критерия, распределение похоже на нормальное по критерию Пирсона (ГОСТ)'
        )
        return True
    else:
        print(
            f'Сумма хи-квадрат {sum_xi} >= {criteries[keys[k+1]]} критерия, распределение не похоже на нормальное по критерию Пирсона (ГОСТ)'
        )
        return False
def print_table(headers, rows):
    print(tabulate(rows, headers=headers, tablefmt="pretty"))


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
    contr_excess = 1 / (excess ** 0.5)
    return excess, contr_excess, assimetry


def split_array_by_range(content: list, max_range):
    result = []
    array = content.copy()
    array = sorted(array)
    while len(array) != 0:
        array = sorted(array)
        result.append([])
        result[len(result) - 1].append(array[0])
        array.pop(0)
        while (
                max(result[len(result) - 1]) - min(result[len(result) - 1]) <= max_range
                and len(array) != 0
        ):
            result[len(result) - 1].append(array[0])
            array.pop(0)
        if (
                max(result[len(result) - 1]) - min(result[len(result) - 1]) > max_range
                and len(result[len(result) - 1]) != 1
        ):
            array.append(result[len(result) - 1][-1])
            result[len(result) - 1].pop(-1)

    if len(result[-1]) == 1:
        result[-2].append(result[-1][0])
        result.remove(result[-1])
    return result


def others_check(sample_data, distributions_d, cexcess, mean):
    alpha = 0.05
    print("\n")
    params_uniform = (
        np.min(sample_data),
        np.max(sample_data),
    )
    # Создаем объект теоретического равномерного распределения
    uniform_dist = uniform(loc=params_uniform[0], scale=params_uniform[1] - params_uniform[0])
    # Сравнение с равномерным распределением
    ks_stat_uniform, p_value_uniform = kstest(sample_data, uniform_dist.cdf)
    print(f'Параметры теста Колмогорова-Смирнова на схожесть с равномерным распределением с масштабом(разницей максимального и минимального значений) {params_uniform[1] - params_uniform[0]}, смещением(минимумом) {np.min(sample_data)}:\nСтатистика = {ks_stat_uniform}, P_value = {p_value_uniform}')
    if p_value_uniform > alpha:
        print(
            "Выборка похожа на равномерное распределение по критерию Колмогорова-Смирнова."
        )
        print(
            f'Контрэксцесс выборки: {cexcess}, контрэксцесс распределения: {distributions_d["Равномерное"]}'
        )
        calculate_confidence_interval(mean, sample_data, "uniform")
    else:
        print(
            "Выборка не похожа на равномерное распределение по критерию Колмогорова-Смирнова."
        )

    # Сравнение с гамма распределением
    a, loc, scale = gamma.fit(sample_data)
    ks_stat_gamma, p_value_gamma = kstest(
        sample_data, gamma(a, loc=loc, scale=scale).cdf
    )
    print(f'Сравнение с теоретическим гамма-распределением с параметрами формы {a}, сдвига {loc}, масштаба {scale}, определенными автоматически программой')
    print(f'Параметры теста Колмогорова-Смирнова на схожесть с гамма-распределением:\nСтатистика = {ks_stat_gamma}, P_value = {p_value_gamma}')
    if p_value_gamma > alpha:
        print("Выборка похожа на гамма-распределение по критерию Колмогорова-Смирнова.")
        print(
            f'Контрэксцесс выборки: {cexcess}, контрэксцесс гамма распределения с параметром формы от 0.5 до 2: {distributions_d["Гамма"]}'
        )
        calculate_confidence_interval(mean, sample_data, "gamma")
    else:
        print(
            "Выборка не похожа на гамма-распределение по критерию Колмогорова-Смирнова."
        )

    # Сравнение с треугольным распределением
    params_triangular = (
        np.min(sample_data),
        np.max(sample_data),
        np.median(sample_data),
    )
    triangular_dist = triang(
        c=(params_triangular[2] - params_triangular[0]) / (params_triangular[1] - params_triangular[0]),
        loc=params_triangular[0], scale=params_triangular[1] - params_triangular[0])

    ks_stat_tri, p_value_tri = kstest(sample_data, triangular_dist.cdf, )
    print(f'Сравнение с теоретическим треугольным распределением с медианой {np.median(sample_data)}, масштабом(разница максимума и минимума) {params_triangular[1] - params_triangular[0]}, минимумом(смещением) {np.min(sample_data)},\nточкой максимума {(params_triangular[2] - params_triangular[0]) / (params_triangular[1] - params_triangular[0])} найденной как отношение разниц медианы и минимума к разнице максимума и минимума')
    print(f'Параметры теста Колмогорова-Смирнова на схожесть с треугольным распределением:\nСтатистика = {ks_stat_tri}, P_value = {p_value_tri}')
    if p_value_tri > alpha:
        print(
            "Выборка похожа на треугольное распределение по критерию Колмогорова-Смирнова."
        )
        print(
            f'Контрэксцесс выборки: {cexcess}, контрэксцесс распределения: {distributions_d["Треугольное"]}'
        )
        calculate_confidence_interval(mean, sample_data, "triangular")
    else:
        print(
            "Выборка не похожа на треугольное распределение по критерию Колмогорова-Смирнова."
        )

    # Сравнение с лапласа распределением
    loc_laplace, scale_laplace = laplace.fit(sample_data)
    ks_stat_laplace, p_value_laplace = kstest(
        sample_data, laplace(loc=loc_laplace, scale=scale_laplace).cdf
    )
    print(f'Сравнение с теоретическим лапласа распределением с сдвигом {loc_laplace}, маcштабом {scale_laplace}')
    print(f'Параметры теста Колмогорова-Смирнова на схожесть с треугольным распределением:\nСтатистика = {ks_stat_laplace}, P_value = {p_value_laplace}')
    if p_value_laplace > alpha:
        print(
            "Выборка похожа на лапласовское распределение по критерию Колмогорова-Смирнова."
        )
        print(
            f'Контрэксцесс выборки: {cexcess}, контрэксцесс распределения: {distributions_d["Лапласово"]}'
        )
        calculate_confidence_interval(mean, sample_data, "laplace")
    else:
        print(
            "Выборка не похожа на лапласовское распределение по критерию Колмогорова-Смирнова."
        )

    plt.figure(figsize=(20, 14))

    # Треугольное распределение
    params_tri = (
        np.min(sample_data),
        np.mean(sample_data),
        np.max(sample_data),
    )  # Используем минимум, среднее и максимум из выборки
    tri_dist = triang(
        c=(params_tri[1] - params_tri[0]) / (params_tri[2] - params_tri[0]),
        loc=params_tri[0],
        scale=params_tri[2] - params_tri[0],
    )
    print(f'Параметры теоретического треугольного распределения, построенного на гистограмме поверх изучаемой выборки: точка максимума {(params_tri[1] - params_tri[0]) / (params_tri[2] - params_tri[0])}, берется как отношение разниц среднего к минимуму и максимума к минимуму, смещение(минимум) {params_tri[0]}, масштаб(разница максимума и минимума) {params_tri[2] - params_tri[0]}')
    # Лапласово распределение
    params_laplace = (
        np.mean(sample_data),
        np.std(sample_data),
    )  # Используем среднее и стандартное отклонение из выборки
    laplace_dist = laplace(*params_laplace)
    print(f'Параметры теоретического лапласова распределения, построенного на гистограмме поверх изучаемой выборки: cреднее значение {np.mean(sample_data)}, СКО {np.std(sample_data)}')

    # Равномерное распределение
    params_uniform = (
        np.min(sample_data),
        np.max(sample_data),
    )  # Используем минимум и максимум из выборки
    print(f'Параметры теоретического равномерного распределения, построенного на гистограмме поверх изучаемой выборки: смещение { np.min(sample_data)}, масштаб(разница максимального с минимальным) {params_uniform[1] - params_uniform[0]}')
    uniform_dist = uniform(
        loc=params_uniform[0], scale=params_uniform[1] - params_uniform[0]
    )
    # Гамма-распределение
    params_gamma = (
        np.mean(sample_data),
        np.std(sample_data),
    )  # Используем среднее и стандартное отклонение из выборки
    print(f'Параметры теоретического гамма распределения, построенного на гистограмме поверх изучаемой выборки: форма {params_gamma[0] ** 2 / params_gamma[1] ** 2} считается как отношение квадратов среднего и СКО,\nмасштаб {params_gamma[1] ** 2 / params_gamma[0]}, считается как отношение квадрата СКО к среднему')
    gamma_dist = gamma(
        a=params_gamma[0] ** 2 / params_gamma[1] ** 2,
        scale=params_gamma[1] ** 2 / params_gamma[0],
    )

    # График сравнения с равномерным распределением
    plt.subplot(2, 4, 5)
    probplot(sample_data, dist=uniform, sparams=(len(content) - 1,), plot=plt)
    plt.title("Сравнение с равномерным распределением")

    # График сравнения с гамма распределением
    plt.subplot(2, 4, 6)
    probplot(sample_data, dist=gamma, sparams=(len(content) - 1,), plot=plt)
    plt.title("Сравнение с гамма распределением")

    # График сравнения с лапласа распределением
    plt.subplot(2, 4, 7)
    probplot(sample_data, dist=laplace, sparams=(len(content) - 1,), plot=plt)
    plt.title("Сравнение с лапласа распределением")

    ax = plt.subplot(2, 4, 8)
    sm.qqplot(np.array(sample_data), dist=triang, fit=True, line="45", ax=ax)
    plt.title("Сравнение с треугольным распределением")

    # Гистограммы
    plt.subplot(2, 4, 4)
    plt.hist(
        sample_data, bins="auto", density=True, alpha=0.7, color="g", label="Выборка"
    )
    x_tri = np.linspace(np.min(sample_data), np.max(sample_data), 1000)
    plt.plot(x_tri, tri_dist.pdf(x_tri), "r", label="Треугольное")
    plt.title("Треугольное распределение")
    plt.legend()

    plt.subplot(2, 4, 3)
    plt.hist(
        sample_data, bins="auto", density=True, alpha=0.7, color="b", label="Выборка"
    )
    x_laplace = np.linspace(np.min(sample_data), np.max(sample_data), 1000)
    plt.plot(x_laplace, laplace_dist.pdf(x_laplace), "r", label="Лапласово")
    plt.title("Лапласово распределение")
    plt.legend()

    plt.subplot(2, 4, 1)
    plt.hist(
        sample_data,
        bins="auto",
        density=True,
        alpha=0.7,
        color="purple",
        label="Выборка",
    )
    x_uniform = np.linspace(np.min(sample_data), np.max(sample_data), 1000)
    plt.plot(x_uniform, uniform_dist.pdf(x_uniform), "r", label="Равномерное")
    plt.title("Равномерное распределение")
    plt.legend()

    plt.subplot(2, 4, 2)
    plt.hist(
        sample_data, bins="auto", density=True, alpha=0.7, color="b", label="Выборка"
    )
    x_gamma = np.linspace(np.min(sample_data), np.max(sample_data), 1000)
    plt.plot(x_gamma, gamma_dist.pdf(x_gamma), "r", label="Гамма")
    plt.title("Гамма-распределение")
    plt.legend()

    plt.tight_layout()
    plt.show()


def trust_borders_normal(data, Sx, t_student):
    if len(data) - 1 > len(t_student):
        epsilon = t_student[31] * Sx
    else:
        epsilon = t_student[len(data) - 1] * Sx
    K = epsilon / Sx
    delta = K * Sx
    return epsilon, delta


def normality_check(data):
    alpha = 0.05
    # Проверка нормальности с помощью критерия Колмогорова-Смирнова
    params_normal = (
        np.mean(data),
        np.std(data),
    )
    # Создаем объект теоретического нормального распределения
    normal_dist = norm(loc=params_normal[0], scale=params_normal[1])
    statistic_ks, p_value_ks = kstest(data, normal_dist.cdf)
    print(f'Сравнение с теоретическим нормальным распределением с отклонением {S}, средним {mean}')
    print(
        f'Параметры теста Колмогорова-Смирнова на схожесть с нормальным распределением:\nСтатистика = {statistic_ks}, P_value = {p_value_ks}')
    if p_value_ks > alpha:
        print(
            "Выборка согласуется с нормальным распределением по критерию Колмогорова-Смирнова."
        )
        return True
    else:
        print(
            "Выборка не согласуется с нормальным распределением по критерию Колмогорова-Смирнова."
        )
        return False


def normality_graphs(data):
    plt.figure(figsize=(12, 6))
    # Гистограмма
    plt.subplot(1, 2, 1)
    plt.hist(data, bins="auto", density=True, alpha=0.7, color="g")
    mu, std = np.mean(data), np.std(data)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = np.exp(-((x - mu) ** 2) / (2 * std ** 2)) / (std * np.sqrt(2 * np.pi))
    plt.plot(x, p, "k", linewidth=2)
    plt.title("Гистограмма и нормальное распределение")

    # Q-Q график
    plt.subplot(1, 2, 2)
    probplot(data, dist="norm", plot=plt)
    plt.title("Q-Q график")

    # Отображение графиков
    plt.tight_layout()
    plt.show()


def read_table(filename):
    df = pd.read_excel(filename + ".xlsx", header=None)
    df.columns = ["A", "B"]
    coefficients_dict = dict(zip(df["A"], df["B"]))
    return coefficients_dict


def calculate_confidence_interval(mean, sample, distribution, alpha=0.05):
    print('Квантиль берется для уровня 1-alpha/2, где alpha=0.05')
    if distribution == "laplace":
        median_sample = np.median(sample)
        # Вычисляем квантиль лапласовского распределения
        a, loc_laplace, scale_laplace = laplace.fit(sample)
        q_laplace = laplace.ppf(1 - alpha / 2,loc=loc_laplace, scale=scale_laplace)
        # Вычисляем погрешность, убеждаясь, что делитель не равен нулю
        n = len(sample)
        scale_laplace = median_sample, np.mean(np.abs(sample - median_sample))
        margin_of_error_laplace = q_laplace * (
            scale_laplace[0] / np.sqrt(n) if n > 0 else 1
        )
        print(f'Квантиль равен {q_laplace} для теоретического лапласовского распределения со смещением {loc_laplace} и масштабом {scale_laplace}.\nПогрешность считается как произведение квантиля на отношение медианы выборки {scale_laplace[0]} к корню размера выборки {np.sqrt(n)}')
        print(
            f"Измеренное значение = {round(mean, 6)} +- {round(margin_of_error_laplace, 6)},\nпогрешность посчитана по квантилю лапласовского распределения\n"
        )

    elif distribution == "uniform":
        min_val, max_val = np.min(sample), np.max(sample)
        a, loc_uni, scale_uni = uniform.fit(sample)
        # Вычисляем квантиль равномерного распределения
        q_uniform = uniform.ppf(1 - alpha / 2, loc=loc_uni, scale=scale_uni)
        margin_of_error_uniform = (
                q_uniform * (max_val - min_val) / (2 * np.sqrt(3) * np.sqrt(len(sample)))
        )
        print(f'Квантиль равен {q_uniform}, считается как квантиль для теоретического равномерного распределения с параметами\nформы {a},смещения {loc_uni}, масштаба {scale_uni}, определенными программой автоматически из данной выборки.\nПогрешность считается как произведение квантиля на отношение разницы максимума и минимума выборки {max_val - min_val} к удвоенному корню из трех помноженному на квадратный корень размера выборки {np.sqrt(len(sample))}')
        print(
            f"Измеренное значение = {round(mean, 6)} +- {round(margin_of_error_uniform, 6)},\nпогрешность посчитана по квантилю равномерного распределения\n"
        )

    elif distribution == "triangular":
        peak_location = np.argmax(np.bincount(sample))
        a, loc_tri, scale_tri2 = triang.fit(sample)
        # Приводим значение к интервалу [0, 1]
        normalized_peak_location = peak_location / (np.max(sample) - np.min(sample))
        # Вычисляем квантиль треугольного распределения
        scale_tri = (np.max(sample) - np.min(sample)) / np.sqrt(6)
        q_triangular = triang.ppf(1 - alpha / 2, c=normalized_peak_location, scale=scale_tri, loc=loc_tri)
        margin_of_error_tri = q_triangular * (scale_tri / np.sqrt(len(sample)))
        print(f'Квантиль равен {q_triangular}, считается как квантиль теоретического треугольного распределения с пиком находящимся {normalized_peak_location}, параметами смещения {loc_tri}, масштабом {scale_tri}\n определенными программой автоматически по данным изучаемой выборки.\nМасштаб выборки по треугольному распределению равна отношению между разницей максимального и минимального отношения деленных на корень из 6 = {scale_tri},\n погрешность считается как произведение квантиля на отношение дисперсии выборки к корню ее размера ({np.sqrt(len(sample))})')
        print(
            f"Измеренное значение = {round(mean, 6)} +- {round(margin_of_error_tri, 6)},\nпогрешность посчитана по квантилю равномерного распределения\n"
        )

    elif distribution == "gamma":
        a_gamma, loc_gamma, scale_gamma = gamma.fit(sample)
        # Вычисляем квантиль гамма распределения
        q_gamma = gamma.ppf(1 - alpha / 2, a_gamma, loc=loc_gamma, scale=scale_gamma)
        margin_of_error_gamma = q_gamma - loc_gamma
        print(f'Квантиль равен {q_gamma}, считается как квантиль теоретического гамма-распределения с параметами\nформы {a_gamma},смещения {loc_gamma}, масштаба {scale_gamma} определенными программой автоматически по данным изучаемой выборки.\nПогрешность считается как произведение квантиля на отношение дисперсии выборки к корню ее размера ({np.sqrt(len(sample))})')

        print(
            f"Измеренное значение = {round(mean, 6)} +- {round(margin_of_error_gamma, 6)},\nпогрешность посчитана по квантилю равномерного распределения\n"
        )


# считывание выборки из текстового файла
def read_chosen_nums(filename):
    f = open(filename + ".txt", "r")
    content = []
    for i in f.readlines():
        if i != "" and i != "\n" and i != " \n":
            content.append(float(i[:-2].replace(",", ".")))
    f.close()
    return content


def read_two_coeff(filename):
    df = pd.read_excel(filename + ".xlsx", header=None)
    df.iloc[:, 2] = df.iloc[:, 2].replace({",": "."}, regex=True)
    df.iloc[:, 1] = df.iloc[:, 1].replace({",": "."}, regex=True)
    coefficients_dict = {}
    for _, row in df.iterrows():
        index_key = int(row.iloc[0])
        coefficients_dict[index_key] = [float(row.iloc[2]), float(row.iloc[1])]
    return coefficients_dict


def find_closest_distribution(variable, distribution_dict):
    closest_value = min(distribution_dict.keys(), key=lambda x: abs(x - variable))
    closest_distribution = distribution_dict[closest_value]
    return closest_value, closest_distribution


if __name__ == "__main__":

    content = read_chosen_nums("input")
    grubbs_criteries = read_table("граббс")
    number_of_freedoms = len(content) - 1
    print(f"Размер выборки: {len(content)}")
    mean, S, Sx = avg_square_etc(content)
    old_len = len(content)
    while check_grubbs_max(content, S, grubbs_criteries, mean):
        print(
            f"Обнаружен выброс максимального значения ({max(content)}) по критерию Граббса, удаляем значение и пересчитываем параметры выборки"
        )
        content.remove(max(content))
        mean, S, Sx = avg_square_etc(content)
    while check_grubbs_min(content, S, grubbs_criteries, mean):
        print(
            f"Обнаружен выброс минимального значения ({min(content)}) по критерию Граббса, удаляем значение и пересчитываем параметры выборки"
        )
        content.remove(min(content))
        mean, S, Sx = avg_square_etc(content)
    if old_len == len(content):
        print("Выбросов по критерию Граббса не обнаружено\n")
    if 15 < len(content) <= 50:
        d_criteries = read_two_coeff("квантили критерий 1")
        p_criteries = read_two_coeff("P для Z")
        z_criterie = read_table("Z от P")

        if check_d_critery(content, mean, d_criteries) and check_second_critery(
                content, S, p_criteries, z_criterie
        ):
            normal1 = True
            print("Распределение принадлежит нормальному по составному критерию ГОСТ")
        else:
            normal1 = False
            print("Распределение не принадлежит нормальному по составному критерию ГОСТ")
        normal2 = normality_check(content)
        if normal2 and normal1:
            print("Cоставной критерий нормальности по ГОСТ и критерий Колмогорова-Смирнова на нормальность пройдены")
        normal = normal1 or normal2
    elif len(content) > 50:
        xi_crit = read_table("хиКритерии")
        normal1 = check_normality_big_nums(content, S, mean, xi_crit)
        normal2 = normality_check(content)
        if normal2 and normal1:
            print("Критерии хи-квадрат и Колмогорова-Смирнова на нормальность пройдены")
        normal = normal1 or normal2
    else:
        print("Выборка слишком малого размера, проверка на нормальность по ГОСТ не предполагается, проверим исключительно по критерию Колмогорова-Смирнова")
        normal = normality_check(content)
    normality_graphs(content)
    t_student = read_table("СтьюдентГОСТ")
    trust_orders, trust_borders_meas = trust_borders_normal(content, Sx, t_student)
    print(f"По условиям задания считается, что НСП=0")
    print(
        f"Доверительные границы случайной погрешности измеряемой величины: {trust_orders}\nДоверительные границы погрешности измеряемой величины: {trust_borders_meas} \n"
    )
    excess, contr_excess, assimetry = find_excess_and_etc(content, S)
    print(f"Эксцесс: {excess}")
    print(f"Контрэксцесс: {contr_excess}")
    print(f"Ассиметрия: {assimetry}")
    distribution_dict = {
        0.408: "Лапласово",
        0.577: "Нормальное",
        0.745: "Равномерное",
        0.645: "Треугольное",
        0.363: "Гамма",
    }
    flipped_dict = {v: k for k, v in distribution_dict.items()}
    closest_value, closest_distribution = find_closest_distribution(
        contr_excess, distribution_dict
    )
    print(
        f"\nКоэксцесс выборки равен {contr_excess:.4f}, наиболее близок к коэкцессу {closest_value:.4f}, который имеет "
        f"{closest_distribution} распределение."
    )
    if not normal:
        others_check(content, flipped_dict, contr_excess, mean)
        print(
            f"Вывод погрешности, посчитанной не по индивидуальному квантилю распределения:\nИзмеренное значение величины равно {round(mean, 6)} +- {round(trust_borders_meas, 6)}, P=0.05"
        )
    else:
        print(
            f"\nЭталонный контрэксцесс нормального распределения равен {flipped_dict['Нормальное']}")
        print(
            f"\nИзмеренное значение величины: {round(mean, 6)}\nСреднеквадратическое отклонение среднего арифметического: {round(Sx, 6)}\nРазмер выборки: {len(content)}\nНСП по условию равно нулю."
        )