from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
import re
from typing import Dict, Union
import math

my_filter_block = np.array([[1, 1, 1, 1, 1],
                            [1, 2, 3, 2, 1],
                            [1, 3, 4, 3, 1],
                            [1, 2, 3, 2, 1],
                            [1, 1, 1, 1, 1]])

my_expand_block = np.array([[0, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 0]])
my_expand_block = my_expand_block / 13

my_kernel = np.array([[2,2,2,2,1],
                      [2,2,2,1,0],
                      [2,2,1,0,0],
                      [2,1,0,0,0],
                      [1,0,0,0,0]])
my_kernel = my_kernel - 1


def nature2d(x, y):
    return np.exp(-(x**2))*np.exp(-(y**2))
# 二维正态分布


n = 5
nature_filter_block = np.ones((n,n))
half_size = n // 2
for i in range(-half_size, half_size+1):
    for j in range(-half_size, half_size+1):
        nature_filter_block[half_size+i][half_size+j] *= nature2d(i*1.5, j*1.5)
# 生成吸引核


def read_image(path):
    image = Image.open(path)
    image = image.convert("L")
    return np.array(image)
    # output is L mode (H,W) 输出为灰度格式，size(H,W)


def show_as_pic(arr):
    pic_image = Image.fromarray(arr)
    pic_image.show()
    return 1


def normalization(arr):
    upper_bound = np.max(arr)
    factor = 255 / upper_bound
    return arr * factor


def filterate(arr, filter_size):
    # 过滤一定强度以下的噪声
    return np.where(arr > filter_size, arr, 0)


def _no_out_of_index(xy, x_range, y_range):
    if xy[0] < x_range[0]:
        xy[0] = x_range[0]
    elif xy[0] > x_range[1]:
        xy[0] = x_range[1]
    if xy[1] < y_range[0]:
        xy[1] = y_range[0]
    elif xy[1] > y_range[1]:
        xy[1] = y_range[1]
    return xy


def _get_att_vector(pad_arr, filter_block, half_h, half_w, x, y, att_factor=.01):
    # 获取吸引力向量
    x += half_h
    y += half_w
    # 调整参数以适应下方操作
    cut_arr = pad_arr[x-half_h:x+(half_h+1), y-half_w:y+(half_w+1)]
    # 截取参考区域
    att_vector = np.array([0, 0])
    for i in range(filter_block.shape[0]):
        for j in range(filter_block.shape[1]):
            att_vector[0] += cut_arr[i][j] * (i-half_h)
            att_vector[1] += cut_arr[i][j] * (j-half_w)
    att_vector = att_vector * att_factor
    att_vector = np.rint(att_vector)
    att_vector = att_vector.astype(int)
    return att_vector


def gather(arr, filter_block, gather_factor=.01):
    # 聚合化
    assert filter_block.shape[0] % 2 == 1
    assert filter_block.shape[1] % 2 == 1
    # 预期过滤器尺寸为奇数
    half_h = filter_block.shape[0] // 2
    half_w = filter_block.shape[1] // 2
    pad_arr = np.pad(arr, (half_h, half_w), mode="constant", constant_values=0)
    # 预处理以协调_get_att_vector
    new_block = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            att_vector = _get_att_vector(pad_arr, filter_block, half_h, half_w, i, j, att_factor=gather_factor)
            target_vector = att_vector + np.array([i, j])
            target_vector = _no_out_of_index(target_vector, (0, arr.shape[0]-1), (0, arr.shape[1]-1))
            new_block[target_vector[0]][target_vector[1]] += arr[i][j]
    # 遍历并逐个聚合所有数据点
    return new_block


def collect(arr):
    # 拾取所有聚合后仍存在的点
    collected_points = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i][j] != 0:
                collected_points.append((i, j))
    return collected_points


def fuzzy_expand(arr, expand_block):
    a = 2
    b = 2
    pad_arr = np.pad(arr, (a, b), mode="constant", constant_values=0)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            # 遍历
            for l1 in range(-a, a+1):
                for l2 in range(-b, b+1):
                    # 应与扩大矩阵匹配
                    pad_arr[i+l1][j+l2] += arr[i][j] * expand_block[l1+2][l2+2]
    return pad_arr[a:arr.shape[0]+a+1, b:arr.shape[1]+b+1]


def how_isolated(arr, points_list):
    num = 0
    pad_arr = np.pad(arr, (5, 5), mode="constant", constant_values=0)
    for xy in points_list:
        for i in range(-5, 6):
            for j in range(-5, 6):
                if pad_arr[xy[0]+i+5][xy[1]+j+5] != 0:
                    num += 1

    return 1 - num / (len(points_list) * 25)


def regional_acc(pad_arr, x, y, block):
    # 预期核尺寸为奇数
    half_h = block.shape[0] // 2
    half_w = block.shape[1] // 2
    result = 0
    for i in range(-half_h, half_h+1):
        for j in range(-half_w, half_w+1):
            result += pad_arr[x+i][y+j]
    return result


def shrink(arr, kernel, shrink_sta):
    t_kernel = kernel.T
    pad_arr = np.pad(arr, (kernel.shape[0]//2, kernel.shape[1]//2), mode="constant", constant_values=0)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            a = regional_acc(pad_arr, i, j, kernel)
            b = regional_acc(pad_arr, i, j, t_kernel)
            if abs(a) > shrink_sta or abs(b) > shrink_sta:
                arr[i][j] /= 2
    return arr


def averaging(arr, size):
    half_size = size // 2
    pad_arr = np.pad(arr, (half_size, half_size), mode="constant", constant_values=0)
    temp_block = np.ones((size, size))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            mean_value = regional_acc(pad_arr, i+half_size, j+half_size, temp_block) / size ** 2
            arr[i][j] = mean_value
    return arr


def sq_normalization(arr):
    arr = normalization(arr)
    arr = arr ** 0.5
    return normalization(arr)


def nature_normalization(arr):
    arr = normalization(arr)
    arr = (1 - np.exp(-arr / 85)) * 255
    return normalization(arr)


def blur(arr, block):
    pad_h = block.shape[0] - (arr.shape[0] % block.shape[0])
    pad_w = block.shape[1] - (arr.shape[1] % block.shape[1])
    new_h = (arr.shape[0] // block.shape[0]) + 1
    new_w = (arr.shape[1] // block.shape[1]) + 1
    step_h = block.shape[0]
    step_w = block.shape[1]

    pad_arr = np.pad(arr, ((pad_h, 0), (pad_w, 0)), mode="constant", constant_values=0)
    new_arr = np.zeros((new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            temp_value = regional_acc(pad_arr, i*step_h+(step_h//2), j*step_w+(step_w//2), block)
            new_arr[i][j] = temp_value

    return new_arr


def _is_next(p1, p2):
    if (int(abs(p1[0] - p2[0])) < 2) and (int(abs(p1[1] - p2[1])) < 2):
        return True
    else:
        return False


def group_make(arr, points_list):
    # 把聚成一团的点结合为整体
    fg_list = np.zeros(len(points_list), dtype=int)
    fg_ord = 1
    is_iso = True
    groups = []
    value_list = []
    for i in range(len(points_list)):
        is_iso = True
        for j in range(i):
            if _is_next(points_list[i], points_list[j]):
                groups[fg_list[j]-1].append(points_list[i])
                fg_list[i] = fg_list[j]
                value_list[fg_list[j]-1] += arr[points_list[i][0], points_list[i][1]]
                is_iso = False
                break
        if is_iso:
            fg_list[i] = fg_ord
            groups.append([points_list[i]])
            value_list.append(arr[points_list[i][0], points_list[i][1]])
            fg_ord += 1
    return np.array(value_list), groups


def groups_to_points(arr, groups):
    points = []
    for i in groups:
        x = 0
        y = 0
        ssum = 0
        for j in range(len(i)):
            x += i[j][0] * arr[i[j][0]][i[j][1]]
            y += i[j][1] * arr[i[j][0]][i[j][1]]
            ssum += arr[i[j][0]][i[j][1]]
        x, y = x / ssum, y / ssum
        points.append([x, y])
    return np.array(points)


def points_list_to_arr(ori_arr, points_list, value_list):
    new_arr = np.zeros_like(ori_arr)

    for i in range(len(points_list)):
        new_arr[int(points_list[i][0])][int(points_list[i][1])] = value_list[i]
    return new_arr


def point_extraction(path):
    pic_arr = read_image(path)

    pic_arr = sq_normalization(pic_arr)
    pic_arr = filterate(pic_arr, 150)
    pic_arr = blur(pic_arr, np.ones((5, 5)))

    points = collect(pic_arr)
    group_value, points_groups = group_make(pic_arr, points)
    group_points = groups_to_points(pic_arr, points_groups)

    return group_points, group_value


def _get_fourier_value(x_list, wavelength):
    return max(abs(sum(np.sin(x_list * 2 * np.pi / wavelength))), abs(sum(np.cos(x_list * 2 * np.pi / wavelength))))


def _get_peak_value(x_list):
    peak_value_list = []
    peak_ord_list = []
    for i in range(1, len(x_list) - 1):
        if x_list[i] > x_list[i-1] and x_list[i] >= x_list[i+1]:
            peak_value_list.append(x_list[i])
            peak_ord_list.append(i)
    if x_list[0] > x_list[1] and x_list[0] >= x_list[len(x_list)-1]:
        peak_value_list.append(x_list[0])
        peak_ord_list.append(0)
    if x_list[len(x_list)-1] >= x_list[len(x_list)-2] and x_list[len(x_list)-1] > x_list[0]:
        peak_value_list.append(x_list[len(x_list)-1])
        peak_ord_list.append(len(x_list)-1)

    return np.array(peak_value_list), np.array(peak_ord_list)


def fourier(point_list, rate, percentage):
    # 数据点，分辨率系数， 范围系数
    x_list = point_list[:, 0]
    y_list = point_list[:, 1]
    x_range = np.max(x_list) - np.min(x_list)
    y_range = np.max(y_list) - np.min(y_list)
    x_result_list = []
    y_result_list = []
    for i in range(rate):
        x_result = _get_fourier_value(x_list, (i + 1) * (x_range / rate) * (percentage / 100))
        x_result_list.append(x_result)
        y_result = _get_fourier_value(y_list, (i + 1) * (y_range / rate) * (percentage / 100))
        y_result_list.append(y_result)
    return np.array(x_result_list), np.array(y_result_list)


def wash_fourier(f_list):
    _, ord_list = _get_peak_value(f_list)
    # 获取所有极值的存储id
    washed_list = []
    for i in range(len(f_list)):
        if i in ord_list:
            washed_list.append(f_list[i])
        else:
            washed_list.append(0)

    return np.array(washed_list)


def _match_func(v_list, p_list, a, sep):
    return sum(
        (min(((p_list-a)/sep)**2) +
         min(((p_list-a/2)/sep)**2) +
         min(((p_list-a/3)/sep)**2) +
         min((p_list-a/4)**2)/sep) *
        v_list /
        (a*sep)**2
    )


def match_fourier(f_list, sep):
    new_order_list = []
    val_list, ord_list = _get_peak_value(f_list)
    for i in range(len(f_list)):
        new_order_list.append(_match_func(val_list, ord_list, i, sep))
    return np.array(new_order_list)


def wash_matched(f_list):
    f_list = -f_list
    f_list = wash_fourier(f_list)
    return -f_list


def _inner_product(points_list, base):
    # 获取点集对特定矢量的内积，输出为对应点内积组成的数组
    result = []
    for i in points_list:
        result.append(i[0]*base[0]+i[1]*base[1])
    return np.array(result)


def _scan_by_direct(points_list, rate):
    record_list = []
    sep = np.pi / rate
    for i in range(rate):
        base_vector = (np.cos(i * sep), np.sin(i * sep))
        record_list.append(_inner_product(points_list, base_vector))
        # 返回序数
    return np.array(record_list)


# 此处存在一个疑似bug
def _get_poly_degree_by_ord(ord, value_list):
    # 获取点集的聚集程度，返回值为其聚集度
    sum = 0
    k = (np.max(value_list) - np.min(value_list))*0.01
    base_x = value_list[ord]
    for i in range(len(value_list)):
        if not (i == ord):
            x = value_list[i]
            sum += np.exp(-((x-base_x)/k)**2)
    deno = np.pi ** 0.5 * len(value_list) * 0.001
    # deno为表征所有元素离散存在时的聚集度，用于对输出值进行归一化
    return sum/deno


def get_direct_ploy_degrees(points_list, rate):
    direct_in_list = _scan_by_direct(points_list, rate)
    result_list = []
    for i in direct_in_list:
        ssum = 0
        for j in range(len(i)):
            ssum += _get_poly_degree_by_ord(j, i)
        result_list.append(ssum)
        # 返回序数列
    return result_list


def top_n_indices(sequence, n):
    # 将元素和索引一起存储
    indexed_elements = [(value, index) for index, value in enumerate(sequence)]

    # 按元素值降序排序
    sorted_elements = sorted(indexed_elements, key=lambda x: (-x[0], x[1]))

    # 提取前n个元素的索引
    top_indices = [elem[1] for elem in sorted_elements[:n]]

    # 按原始数列中的出现顺序返回
    return sorted(top_indices)


def _get_normal_vector_effect_list(p_list, base, sep, rate):
    ssum_list = []
    for i in range(rate):
        v_list = _inner_product(p_list, base*sep*i)
        # 获取的点阵与层向量内积值列表，i值对应长度为sep*i的向量
        minus = min(v_list)
        v_list -= minus
        # 移动原点位置便于比较
        ssum_list.append(sum(np.exp(-((v_list-1)*10)**2)))
        # 使用过滤函数捕获点列内积与1相交数据，图线出现峰值意味着有有一列点与1线相交
    return np.array(ssum_list)


def get_max_effective_vector(point_list, base_degree, sep, rate):
    base = np.array([np.cos(base_degree), np.sin(base_degree)])
    v_list = _get_normal_vector_effect_list(point_list, base, sep, rate)
    # 获取遍历的法向量效用列表，列表值为最接近线间距倒数的值
    max_v = max(v_list)
    for i in range(rate):
        if (v_list[rate-i-1] > 0.5) and (v_list[rate-i-1] > v_list[rate-i-2]):
            return (rate-i-1)*sep
    # 提取列表中最大的充分大的项，用于衡量满足条件的最大值
    return -1


def get_main_vector(scale, rate, path):
    # 获取主向量
    # 像素比例尺(nm-1/像素)，分辨率(至少1000所得结果才具有可信度)
    points, _ = point_extraction(path)
    # 返回点坐标1:7收缩
    # points为真距离数据

    y = get_direct_ploy_degrees(points, rate)
    x = np.arange(0, len(y))/len(y)*180
    peak_values, peak_orders = _get_peak_value(y)
    picked = peak_orders[top_n_indices(peak_values, 2)] * np.pi / rate
    # picked为真角度数据（弧度）

    vector_length = []
    for i in range(len(picked)):
        vector_length.append(get_max_effective_vector(points, picked[i], 0.001, rate))
    vector_length = np.array(vector_length)
    pl_distance = 1/vector_length*5
    # pl_distance为真像素面间距，对应小基矢面

    degree_gap = abs(picked[0] - picked[1])
    leg_length = []
    leg_length.append(pl_distance[0] / np.sin(degree_gap))
    leg_length.append(pl_distance[1] / np.sin(degree_gap))
    return np.array(leg_length) * scale, degree_gap


def extract_cif_data(file_path: str) -> Dict[str, Union[float, str]]:
    # 初始化结果字典
    cif_data = {
        # 晶胞常数
        "a": None,
        "b": None,
        "c": None,
        "alpha": None,
        "beta": None,
        "gamma": None,
        "cell_volume": None,
        # 空间群信息
        "space_group_IT_number": None,
        "space_group_HM": None,
        "space_group_Hall": None,
        "cell_setting": None,
    }

    # 正则表达式匹配CIF文件中的键值对
    patterns = {
        "a": re.compile(r"_cell_length_a\s+(\d+\.\d+\(?\d*\)?)"),
        "b": re.compile(r"_cell_length_b\s+(\d+\.\d+\(?\d*\)?)"),
        "c": re.compile(r"_cell_length_c\s+(\d+\.\d+\(?\d*\)?)"),
        "alpha": re.compile(r"_cell_angle_alpha\s+(\d+\.?\d*)"),
        "beta": re.compile(r"_cell_angle_beta\s+(\d+\.?\d*)"),
        "gamma": re.compile(r"_cell_angle_gamma\s+(\d+\.?\d*)"),
        "cell_volume": re.compile(r"_cell_volume\s+(\d+\.\d+)"),
        "space_group_IT_number": re.compile(r"_space_group_IT_number\s+(\d+)"),
        "space_group_HM": re.compile(r"_symmetry_space_group_name_H-M\s+['\"]?([^'\"]+)['\"]?"),
        "space_group_Hall": re.compile(r"_symmetry_space_group_name_Hall\s+['\"]?([^'\"]+)['\"]?"),
        "cell_setting": re.compile(r"_symmetry_cell_setting\s+['\"]?([^'\"]+)['\"]?"),
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

            for key, pattern in patterns.items():
                match = pattern.search(content)
                if match:
                    value = match.group(1)
                    # 处理带括号的值（如3.5070(2) -> 3.5070）
                    if '(' in value:
                        value = value.split('(')[0]
                    # 转换为浮点数（如果是晶胞常数）
                    if key in ["a", "b", "c", "alpha", "beta", "gamma", "cell_volume"]:
                        cif_data[key] = float(value)
                    else:
                        cif_data[key] = value.strip("'\"")  # 去除可能的引号
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到。")
    except Exception as e:
        print(f"解析CIF文件时出错：{e}")

    return cif_data


def is_extinct(info, h, k, l):
    stile = info['style']
    if stile == 'P':
        return False
    elif stile == 'I':
        if (h + k + l) % 2 == 1:
            return True
        return False
    elif stile == 'F':
        if (h % 2) == (k % 2) and (h % 2) == (l % 2):
            return False
        return True
    elif stile == 'C':
        if (h + k) % 2 == 1:
            return True
        return False
    elif stile == 'A':
        if (k + l) % 2 == 1:
            return True
        return False
    elif stile == 'B':
        if (h + l) % 2 == 1:
            return True
        return False
    elif stile == 'R':
        if (- h + k +l) % 3 == 0:
            return False
        return True


def _get_direction_length(axes, x, y, z):
    q2 = x**2*axes['a']**2 + y**2*axes['b']**2 + z**2*axes['c']**2
    q2 += 2*y*z*np.cos(axes['al']) + 2*x*z*np.cos(axes['be']) + 2*x*y*np.cos(axes['ga'])
    return q2**0.5


def get_close_direction(axes, *args, para_range=5, tolerance=0.05):
    # axes为规定的晶胞常数字典，arg存储待求晶向长度v1, v2, v3
    temp = [[]]*len(args)
    for i in range(-para_range, para_range+1):
        for j in range(-para_range, para_range+1):
            for k in range(-para_range, para_range+1):
                if is_extinct(axes, i, j, k):
                    continue
                l = _get_direction_length(axes, i, j, k)
                for vec_ord in range(len(args)):
                    mis = abs(args[vec_ord]-l)/args[vec_ord]
                    if mis < tolerance:
                        temp[vec_ord].append([i, j, k, mis])
                    else:
                        pass
    if len(temp[0]) == 0 or len(temp[1]) == 0:
        return None
    return temp


def _get_direction_degree(axes, d1, d2):
    inner = axes['a']**2*d1[0]*d2[0] + axes['b']**2*d1[1]*d2[1] + axes['c']**2*d1[2]*d2[2]
    inner += axes['a'] * axes['b'] * np.cos(axes['ga']) * (d1[0] * d2[1] + d1[1] * d2[0])
    inner += axes['a'] * axes['c'] * np.cos(axes['be']) * (d1[0] * d2[2] + d1[2] * d2[0])
    inner += axes['c'] * axes['b'] * np.cos(axes['al']) * (d1[2] * d2[1] + d1[1] * d2[2])
    inner /= (_get_direction_length(axes, d1[0], d1[1], d1[2]))
    inner /= (_get_direction_length(axes, d2[0], d2[1], d2[2]))
    if inner > 1:
        return 0
    if inner < -1:
        return np.pi
    return np.arccos(inner)


def _is_legal_direction(axes, degree, cry_directions, tolerance):
    # cry_direction为数组，存储两个数组，用于存储待测向量组
    mis = abs(degree - _get_direction_degree(axes, cry_directions[0], cry_directions[1])) / degree
    # 未排除非初基情况
    if mis < tolerance:
        return mis
    else:
        return -1


def where_legal_direction(axes, expected_degree, test_direction_group, tolerance=0.05):
    if test_direction_group is None:
        return None
    legal_direction_group = []
    for i in test_direction_group[0]:
        for j in test_direction_group[1]:
            mis = _is_legal_direction(axes, expected_degree, [i, j], tolerance)
            if mis == -1:
                pass
            else:
                legal_direction_group.append([i, j, mis])
                # i,j结构为[h, k, l, mis_l]含长度误差
    if len(legal_direction_group) == 0:
        return None
    else:
        return legal_direction_group


def trans_to_axes(info):
    a = info['a']
    b = info['b']
    c = info['c']
    al = info['alpha'] * np.pi / 180
    be = info['beta'] * np.pi / 180
    ga = info['gamma'] * np.pi / 180

    if be == (np.pi/2):
        y = 0
    else:
        y = np.cos(be) / np.sin(ga) - np.cos(al)
    z = np.cos(al)**2 + y**2
    z = 1 - z
    z = z**0.5
    v = z*a*b*np.sin(ga)*c
    be1 = np.pi - np.arccos(y/np.sin(al))
    if al == (np.pi/2):
        y = 0
    else:
        y = np.cos(al)/np.sin(ga) - np.cos(be)
    al1 = np.pi - np.arccos(y/np.sin(be))
    if ga == (np.pi/2):
        y = 0
    else:
        y = np.cos(ga)/np.sin(al) - np.cos(be)
    ga1 = np.pi - np.arccos(y/np.sin(be))

    c1 = a*b/v
    a1 = b*c/v
    b1 = a*c/v

    axes = {'a': a1, 'al': al1, 'b': b1, 'be': be1, 'c': c1, 'ga': ga1, 'style': info['space_group_HM'][0]}

    return axes


def find_min_indices(arr, n):
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    if n == 0:
        return []
    # 将数组元素与其索引组合成元组列表
    indexed_elements = list(enumerate(arr))
    # 根据元素值排序，若值相同则按索引排序
    sorted_elements = sorted(indexed_elements, key=lambda x: (x[1], x[0]))
    # 确定有效的n值，防止超出数组长度
    valid_n = min(n, len(sorted_elements))
    # 提取前n个元素的索引
    result = [element[0] for element in sorted_elements[:valid_n]]
    return result


def select_best_direction(acceptable_direction, n):
    mis_list = []
    for d in acceptable_direction:
        mis_list.append(((d[0][3])**2 + (d[1][3])**2 + (d[2])**2)**0.5)
    n_list = find_min_indices(acceptable_direction, n)
    result_list = []
    for i in n_list:
        a = [acceptable_direction[i][0][0: 3], acceptable_direction[i][1][0: 3]]
        result_list.append(a)
    return result_list, n_list


def get_natural_direction(d1, d2):
    result = [d1[1] * d2[2] - d1[2] * d2[1], -d1[0] * d2[2] + d1[2] * d2[0], d1[0] * d2[1] - d1[1] * d2[0]]

    gcd = math.gcd(result[0], math.gcd(result[1], result[2]))
    for i in range(3):
        result[i] = int(result[i]/gcd)

    return result


def get_nd(best_directions):
    # 获取观察晶向
    a = []
    for i in best_directions:
        a.append(get_natural_direction(i[0], i[1]))
    return np.array(a)


def dedup(ori_array):
    new_list = []
    for element in ori_array:
        # 检查元素是否已经在new_list中
        found = False
        for item in new_list:
            if np.array_equal(element, item) or np.array_equal(element, -item):
                found = True
                break
        if not found:
            if element[0] < 0:
                new_list.append(-element.copy())
            else:
                new_list.append(element.copy())  # 使用copy()避免引用原始数组
    return np.array(new_list)


def main():
    vector_len, vector_degree = get_main_vector(1., 2000, path="D:\\my_pics\\1001663_001.png")
    print(vector_len, vector_degree * 180 / np.pi)

    cam_constant = 78.125/0.1942
    pix_size = 1.
    scale_factor = 1 / cam_constant * pix_size
    vector_len *= scale_factor
    print(vector_len)

    info = extract_cif_data("C:\\Users\\15194\\Downloads\\1001663.cif")
    axes = trans_to_axes(info)
    possible_direction = get_close_direction(axes, vector_len[0], vector_len[1], tolerance=0.05)
    acceptable_direction = where_legal_direction(axes, vector_degree, possible_direction, tolerance=0.05)

    if acceptable_direction is None:
        print("not found")
    else:
        result, _ = select_best_direction(acceptable_direction, len(acceptable_direction))
        print(dedup(get_nd(result)))


if __name__ == "__main__":
    main()

