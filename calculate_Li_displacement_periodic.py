from pymatgen.io.vasp.outputs import Xdatcar
from pymatgen.core.structure import Structure
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import shutil
from pymatgen.io.vasp.inputs import Poscar

xdatcar = Xdatcar("../XDATCAR_20ps_600K")
structures = xdatcar.structures
result_peak = []

def get_displacement_diagram (index: int, vmax: float = 2.5, show_figure: bool = True):
    index_Li = index
    print(structures[0][index_Li])

    # structures_test = structures[0:2]

    result = []

    for i in range(0,len(structures) - 1001,10):

        displacements = []

        # 当前结构之后50个结构 (100 fs)的 PeriodicSite
        for j in range(i + 10, i + 1001, 10):
            structures[i].append("H", structures[j][index_Li].coords, True) # 添加
            displacement = structures[i][index_Li].distance(structures[i][-1])
            structures[i].remove_sites([len(structures) - 1])   # 删除
            displacements.append(displacement)
            if (j == i + 500):
                result_peak.append(displacement)
        result.append(displacements)


    result = np.array(result)

    if show_figure:
        # 定义自定义颜色映射
        colors = [(0.0, "#FFFFFF"),  # 白色
                  (0.5, "#FF8989"),  # 浅色
                  (1.0, "#D00000")]  # 深色

        custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)


        # 绘制使用自定义颜色映射的网格图
        plt.rcParams['font.family'] = 'Arial'
        fig = plt.figure(figsize=(12, 2))
        im = plt.imshow(result.T, cmap=custom_cmap, aspect='auto', origin='lower',vmin=0, vmax=vmax)

        # 自定义横纵坐标的数值和刻度范围
        original_ticks_x = np.arange(0, 901, 100)  # 原始数据索引
        scaled_labels_x = original_ticks_x * 0.02      # 等比例放大后的标签

        original_ticks_y = np.arange(0, 101, 50)
        scaled_labels_y = original_ticks_y * 0.02

        plt.xticks(ticks=original_ticks_x, labels=scaled_labels_x, fontsize=14)
        plt.yticks(ticks=original_ticks_y, labels=scaled_labels_y, fontsize=14)

        # 设置横纵坐标标签的大小
        plt.xlabel('T (ps)',fontsize=18)
        plt.ylabel('dt (ps)',fontsize=18)

        # 设置colorbar
        # cbar = plt.colorbar(im)
        # cbar.set_label('Distance (Å)', fontsize=18)

        # 设置colorbar的刻度范围和标签的字体大小
        # cbar_ticks = np.linspace(0, 2.5, num=2)
        # cbar.set_ticks(cbar_ticks)
        # cbar.ax.tick_params(labelsize=14)  # 设置colorbar标签字体大小

        fig.subplots_adjust(bottom=0.3)
        plt.tight_layout()  # 调整布局以防止部分标签被裁剪

        plt.show()

def get_peak_plot ():

    with open("result_peak.txt", "w") as f:
        for i in result_peak:
            f.write(f"{i}\n")

    result_peak_np = np.array(result_peak)

    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize=(12, 2))

    print(len(result_peak_np))

    # 生成横坐标（数组元素的下标）
    x = np.arange(len(result_peak_np))

    plt.plot(x, result_peak_np)

    # 自定义横纵坐标的数值和刻度范围
    original_ticks_x = np.arange(0, 901, 100)  # 原始数据索引
    scaled_labels_x = original_ticks_x * 0.02  # 等比例放大后的标签

    # original_ticks_y = np.arange(0, 101, 50)
    # scaled_labels_y = original_ticks_y * 0.02

    plt.xticks(ticks=original_ticks_x, labels=scaled_labels_x, fontsize=14)
    plt.xlim(0, 900)  # 只显示刻度为0到900的数据
    plt.yticks(fontsize=14)

    # 添加标题和标签
    # plt.title('Line Plot of 1D Array')
    plt.xlabel('T (ps)', fontsize=18)
    plt.ylabel('Distance (Å)', fontsize=18)

    # 添加一条 y=2.5 的红色虚线
    plt.axhline(y=2.5, color='red', linestyle='--')

    plt.tight_layout()  # 调整布局以防止部分标签被裁剪

    # 显示图像
    plt.show()

def get_neighbor_Cl (strucs: list[Structure], index_Li: int, if_export_snapshot: bool = False):

    neigh_Cl_sum = []

    for struc in strucs:
        # print(struc[index_Li].frac_coords)
        neigh = struc.get_neighbors(struc[index_Li], 3)
        neigh_Cl = [periodicNeighbor.index for i, periodicNeighbor in enumerate(neigh) if periodicNeighbor.species_string == "Cl"]
        neigh_Cl_sum.append(neigh_Cl)

        # struc.clear()

    num = 169
    for i in neigh_Cl_sum:
        print(f"{num}---{i}")
        num += 1

    if if_export_snapshot:
        for struc in strucs:
            neigh = struc.get_neighbors(struc[index_Li], 3)
            neigh_Cl = [periodicNeighbor.index for i, periodicNeighbor in enumerate(neigh) if
                        periodicNeighbor.species_string == "Cl"]
            neigh_Cl.append(index_Li)  # 加上Li
            remove_atoms = list(set(range(len(struc))) - set(neigh_Cl))

            struc.remove_sites(remove_atoms)

        if not os.path.exists('snapshot'):
            os.mkdir('snapshot')
        elif os.path.exists('snapshot'):
            shutil.rmtree('snapshot')
            os.mkdir('snapshot')

        i = 169
        for structure in strucs:
            poscar = Poscar(structure)
            poscar.write_file('snapshot/structure_{:0>3d}.vasp'.format(i))
            i += 1


def get_Li_Cl_bonds_distribution (strucs: list[Structure], index_Li: int, indices_Cl: list[int]):
    result = []
    for struc in strucs:
        distances = []
        for index_Cl in indices_Cl:
            distance = struc[index_Li].distance(struc[index_Cl])
            distances.append(distance)
        result.append(distances)

    num = 169
    for i in result:
        print(f"{i}")
        num += 1

def get_Li_Cl_bonds_distribution_fixed (strucs: list[Structure], index_Li: int, indices_Cl: list[int]):

    # 参照结构
    ini_struc = strucs[0]

    result = []
    for struc in strucs:
        distances = []
        ini_struc.append("H", struc[index_Li].coords, True)  # 把新结构的Li添加到参照结构中

        for index_Cl in indices_Cl:
            distance = ini_struc[-1].distance(ini_struc[index_Cl]) # 计算距离
            distances.append(distance)

        ini_struc.remove_sites([len(structures) - 1])  # 删除
        result.append(distances)

    num = 169
    for i in result:
        print(f"{i}")
        num += 1

def get_triangle_area (strucs: list[Structure], indices_Cl: list[int], show_figure: bool = False):
    result = []

    for struc in strucs:
        A = struc[indices_Cl[0]].coords
        B = struc[indices_Cl[1]].coords
        C = struc[indices_Cl[2]].coords

        # 计算向量AB和AC
        AB = B - A
        AC = C - A

        # 计算向量AB和AC的叉积
        cross_product = np.cross(AB, AC)

        # 计算叉积的模长
        area = np.linalg.norm(cross_product) / 2

        result.append(area)

    # num = 169
    for i in result:
        print(f"{i}")
        # num += 1


    if show_figure:
        result = np.array(result)
        plt.rcParams['font.family'] = 'Arial'
        plt.figure(figsize=(10, 5))


        # 生成横坐标（数组元素的下标）
        x = np.arange(len(result))
        plt.plot(x, result, color='Red', marker='o', linestyle='-')

        plt.plot(range(3), result[:3], color='red', marker='o', linestyle='-')

        # 自定义横纵坐标的数值和刻度范围
        original_ticks_x = np.arange(0, 16, 3)  # 原始数据索引
        scaled_labels_x = original_ticks_x * 0.02  # 等比例放大后的标签

        # original_ticks_y = np.arange(0, 101, 50)
        # scaled_labels_y = original_ticks_y * 0.02

        plt.xticks(ticks=original_ticks_x, labels=scaled_labels_x, fontsize=14)
        plt.xlim(-1, 16)  # 只显示刻度为0到15的数组
        plt.ylim(0,8)

        # plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # 添加标题和标签
        # plt.title('Line Plot of 1D Array')
        plt.xlabel('T (ps)', fontsize=18)
        plt.ylabel('Area (Å2)', fontsize=18)

        plt.tight_layout()  # 调整布局以防止部分标签被裁剪

        # 显示图像
        plt.show()



# get_displacement_diagram(0, show_figure=True)
# get_peak_plot()


structures_fragment = structures[0:len(structures) - 1001:10]
# get_neighbor_Cl(strucs=structures_fragment[169:185], index_Li=1)
# get_Li_Cl_bonds_distribution(strucs=structures_fragment[169:185], index_Li=1, indices_Cl=[49, 45, 46])
# get_Li_Cl_bonds_distribution_fixed(strucs=structures_fragment[169:185], index_Li=1, indices_Cl=[49, 45, 46])
get_triangle_area(strucs=structures_fragment[169:185], indices_Cl=[49, 45, 46], show_figure=True)

