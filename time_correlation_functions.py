from pymatgen.io.vasp.outputs import Xdatcar
from pymatgen.core.structure import Structure
import lattice_utils
import pandas as pd
import numpy as np
import time
import math
import csv
import matplotlib.pyplot as plt

xdatcar = Xdatcar("../calculate_Cl_displacement/XDATCAR_20ps_600K_LLC")
structures = xdatcar.structures

ini_strucutre = structures[0]

def get_center_ligand_mapping(strucutre : Structure, center : str, ligand : str, r : int = 3):
    '''
    note 确定每个中心阳离子对应的阴离子编号
    strucutre : 晶体结构 (Structure)
    center : 中心阳离子种类 (str)
    ligand : 配位阴离子种类 (str)
    r : 距离阈值 (int, 默认为3)
    return : [[center,[ligand1,ligand2]],[],[]]
    '''
    center_indices = [i for i, site in enumerate(strucutre) if site.species_string == center]
    result = []
    for i in center_indices:
        mapping = []
        neigh = strucutre.get_neighbors(strucutre[i],r)
        neigh_ligand = [periodicNeighbor.index for periodicNeighbor in neigh if periodicNeighbor.species_string == ligand]
        mapping.append(i)
        mapping.append(neigh_ligand)
        result.append(mapping)
    return result

def get_angle_time_correlation_functions(structures: list[Structure], center_index: int, ligand_index: int, time_step: int, step_skip: int, dt=100, save_to_csv=False):
    '''
    note 计算角度时间关联函数
    structures: 晶体结构列表 (list[Structure])
    center_index: 中心阳离子编号
    ligand_index: 配位阴离子编号
    time_step * step_skip: 相邻结构之间的时间差Δt
    dt: 时间平均所取间隔
    save_to_csv: 是否输出结果
    return: Ct, list(data_grid)
    '''

    center_species = structures[0][center_index].species_string
    ligand_species = structures[0][ligand_index].species_string

    # 晶格基矢
    basis_matrix = structures[0].lattice.matrix

    # 数据网格 (时间 fs) 如 [2,4,6,...,1998] 取不到2000，除非加一个POSCAR
    data_grid = range(time_step * step_skip, len(structures) * time_step * step_skip, time_step * step_skip)

    # 时间平均函数 (从time_step * step_skip开始)
    Ct = []

    start_time = time.time()  # 记录循环开始时间
    # 计算时间平均函数
    for t in data_grid:

        dot_product_sum = 0;
        num_time_intervals = 0

        # 每帧结构之间的时间差 Δt
        deta_t = time_step * step_skip
        struc_step = int(t / deta_t)

        # 时间平均
        # dt' = 1个Δt 或者 10个Δt
        for i in range(0, len(structures) - struc_step, dt):

            # t'时刻 中心阳离子原子坐标
            center_coord = structures[i][center_index].coords
            # t' + t时刻 中心阳离子原子坐标
            center_coord_last = structures[i + struc_step][center_index].coords

            # t'时刻 配位阴离子坐标 (考虑周期性, 与中心阳离子最近)
            ligand_coord = structures[i][ligand_index].coords
            ligand_coord = lattice_utils.get_near_periodic_point(center_coord, ligand_coord, basis_matrix)

            # t'时刻 中心阳离子到配位阴离子的单元向量
            vector = ligand_coord - center_coord
            unit_vector = get_unit_vector(vector)

            # t' + t时刻 配位阴离子坐标 (考虑周期性, 与中心阳离子最近)
            ligand_coord_last = structures[i + struc_step][ligand_index].coords
            ligand_coord_last = lattice_utils.get_near_periodic_point(center_coord_last, ligand_coord_last, basis_matrix)

            # t' + t时刻 中心阳离子到配位阴离子的单元向量
            vector_last = ligand_coord_last - center_coord_last
            unit_vector_last = get_unit_vector(vector_last)

            # 单位向量点积 (余弦值) 与 dt'乘积
            dot_product = np.dot(unit_vector, unit_vector_last)

            dot_product_sum += dot_product
            num_time_intervals += 1

        if(num_time_intervals == 0):
            raise Exception("边界条件出错")

        Ct.append(dot_product_sum / num_time_intervals)

        # 每t = time_step * 1000 打印一次
        if (t % (time_step * 1000) == 0):
            current_time = time.time()  # 记录循环结束时间
            elapsed_time = current_time - start_time  # 计算执行时间

            # 输出当前循环次数和执行时间，并立即刷新输出缓冲区
            print(f"Loop {t} fs: Time taken = {elapsed_time:.2f} seconds", flush=True)

    if save_to_csv:
        # 创建一个DataFrame
        df = pd.DataFrame({
            't (fs)': data_grid,
            'C(t)': Ct
        })

        # 将DataFrame写入CSV文件
        df.to_csv(f'C(t)_{center_species}{center_index}-{ligand_species}{ligand_index}.csv', index=False)

    return Ct, list(data_grid)

def get_unit_vector(vector):
    magnitude = np.linalg.norm(vector)  # 计算向量的模长
    if magnitude == 0:
        raise ValueError("零向量没有定义单元向量")
    return vector / magnitude  # 归一化

# note 计算位移时间关联函数
# def get_displacement_time_correlation_functions():



# 计算给定晶胞中两个原子的坐标（考虑周期性）
def get_distance(structures: list[Structure], center_index: int, ligand_index: int):
    # 晶格基矢
    basis_matrix = structures[0].lattice.matrix

    distances = []

    for structure in structures:
        center_coord = structure[center_index].coords
        ligand_coord = structure[ligand_index].coords
        ligand_coord = lattice_utils.get_near_periodic_point(center_coord, ligand_coord, basis_matrix)
        distance = calculate_distance(center_coord, ligand_coord)
        # distance = structure.get_distance(center_index, ligand_index)
        distances.append(distance)

    return distances

def calculate_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2

    # 计算坐标差的平方
    dx = (x2 - x1) ** 2
    dy = (y2 - y1) ** 2
    dz = (z2 - z1) ** 2

    # 计算距离
    distance = math.sqrt(dx + dy + dz)

    return distance


# 1、将阳离子-阴离子配对信息写入csv文件中 (方便服务器拆分计算每一个cluster)
# center_ligand_mapping = [
#     [12, [36, 24, 60, 48]],
#     [13, [37, 25, 61, 49]],
#     [14, [62, 50, 38, 26]]
# ]
def step1(mapping_filename, center : str, ligand : str):
    center_ligand_mapping = get_center_ligand_mapping(ini_strucutre, center, ligand)
    with open(mapping_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 遍历数据列表，将每个元素展开并写入 CSV 文件
        for item in center_ligand_mapping:
            first_col = item[0]
            other_cols = item[1]
            row = [first_col] + other_cols
            writer.writerow(row)

# 2、计算所有聚阴离子团
def step2_all_clusters(mapping_filename: str):

    # 从csv文件中读取mapping数据
    center_ligand_mapping = []
    with open(mapping_filename, mode='r', newline='') as file:
        reader = csv.reader(file)

        # 遍历每一行，将其添加到列表中
        for row in reader:
            # 将每一行的字符串转换为整数
            formatted_row = [int(item) for item in row]
            center_ligand_mapping.append(formatted_row)


    Ct_sum = []
    is_include_t = False
    for index in range(len(center_ligand_mapping)):
        # 分别计算每一个cluster
        # 第index个cluster
        center_index = center_ligand_mapping[index][0]
        ligand_indices = center_ligand_mapping[index][1:]

        for ligand_index in ligand_indices:
            if(not is_include_t):
                Ct, data_grid= get_angle_time_correlation_functions(structures, center_index, ligand_index, 2, 1, dt=100)
                Ct_sum.append(data_grid)
                Ct_sum.append(Ct)
                is_include_t = True
            else:
                Ct = get_angle_time_correlation_functions(structures, center_index, ligand_index, 2, 1)[0]
                Ct_sum.append(Ct)

        # 输出当前循环次数和执行时间，并立即刷新输出缓冲区
        remain = len(center_ligand_mapping) - index - 1
        print(f"Cluster {index} 计算已完成，剩余 {remain} 个", flush=True)

    with open('Ct_sum.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        # 转置数据，使得每个元素为一列
        transposed_data = zip(*Ct_sum)

        # 写入转置后的数据到CSV文件
        writer.writerows(transposed_data)

# 2、计算单个聚阴离子团
def step2_single_cluster(mapping_filename: str):

    # 从index文件中读取聚阴离子团index
    try:
        # 打开当前目录中的 'index' 文件
        with open('index', 'r') as file:
            # 读取第一行
            first_line = file.readline().strip()

            # 将第一个字符转换为整数
            index = int(first_line)

    except FileNotFoundError:
        print("文件 'index' 不存在，请检查文件路径。")
    except ValueError:
        print("无法将读取的字符转换为整数，请检查文件内容。")

    # 从csv文件中读取mapping数据
    center_ligand_mapping = []
    with open(mapping_filename, mode='r', newline='') as file:
        reader = csv.reader(file)

        # 遍历每一行，将其添加到列表中
        for row in reader:
            # 将每一行的字符串转换为整数
            formatted_row = [int(item) for item in row]
            center_ligand_mapping.append(formatted_row)

    # 分别计算每一个cluster
    # 第index个cluster
    center_index = center_ligand_mapping[index][0]
    ligand_indices = center_ligand_mapping[index][1:]

    Ct_sum = []
    for ligand_index in ligand_indices:
        Ct = get_angle_time_correlation_functions(structures, center_index, ligand_index, 2, 1)
        Ct_sum.append(Ct)

    with open(f'cluster-{index}.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        # 转置数据，使得每个元素为一列
        transposed_data = zip(*Ct_sum )

        # 写入转置后的数据到CSV文件
        writer.writerows(transposed_data)

# 3、画图
def step_3(fig_name: str, Ct_filename= 'Ct_sum.csv'):
    # 读取CSV文件
    file_path = Ct_filename  # 替换为你的CSV文件路径
    data = pd.read_csv(file_path)

    plt.rcParams['font.family'] = 'Arial'
    # 确保读取到正确的数据
    print(data.head())

    # 计算第2列到第49列的行平均值
    data['Average'] = data.iloc[:, 1:].mean(axis=1)

    # 确保平均值计算正确
    print(data['Average'].head())

    # 绘制点线图
    plt.figure(figsize=(8, 4))
    plt.plot(data.iloc[:, 0], data['Average'], marker='o', linestyle='-', color='b')
    plt.xlabel('t (fs)', fontsize=18)
    plt.ylabel('C(t)', fontsize=18)

    # 调整 x 轴和 y 轴刻度标签的字体
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlim(0, 10000)
    plt.ylim(0.9, 1)

    # 上移整个图像
    plt.subplots_adjust(bottom=0.15)  # 调整 top 参数以控制图像上移的幅度

    plt.savefig(f'Ct_{fig_name}.png',dpi=300)

    plt.show()


mapping_filename = 'mapping.csv'
step2_all_clusters(mapping_filename)
# step_3('LNOC')