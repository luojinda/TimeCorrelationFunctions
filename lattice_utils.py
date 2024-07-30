import numpy as np

'''
给定两个点坐标A和B (Cartesian坐标), 得到周期性晶胞中离A最近的B'
'''
def get_near_periodic_point(center_point: list[float], ligand_point: list[float], basis_matrix: list[list[float]]):

    vector_X: float = ligand_point[0] - center_point[0]
    vector_Y: float = ligand_point[1] - center_point[1]
    vector_Z: float = ligand_point[2] - center_point[2]
    vector_cart = [vector_X, vector_Y, vector_Z]
    # 得到分数坐标
    vector_frac = Cart2Frac(vector_cart, basis_matrix)
    ligand_point_frac = Cart2Frac(ligand_point, basis_matrix)

    near_point = [None]*3
    if(vector_frac[0] > 0.5):
        near_point[0] = ligand_point_frac[0] - 1
    elif(vector_frac[0] < 0 and -vector_frac[0] > 0.5):
        near_point[0] = ligand_point_frac[0] + 1
    else:
        near_point[0] = ligand_point_frac[0]

    if (vector_frac[1] > 0.5):
        near_point[1] = ligand_point_frac[1] - 1
    elif (vector_frac[1] < 0 and -vector_frac[1] > 0.5):
        near_point[1] = ligand_point_frac[1] + 1
    else:
        near_point[1] = ligand_point_frac[1]

    if (vector_frac[2] > 0.5):
        near_point[2] = ligand_point_frac[2] - 1
    elif (vector_frac[2] < 0 and -vector_frac[2] > 0.5):
        near_point[2] = ligand_point_frac[2] + 1
    else:
        near_point[2] = ligand_point_frac[2]

    # 把nearPoint转换为Cart坐标
    near_point = Frac2Cart(near_point, basis_matrix)

    return near_point

'''
给定两个点坐标A和B (Frac坐标), 得到周期性晶胞中离A最近的B'
'''
def get_near_periodic_point_v2(center_point: list[float], ligand_point: list[float]):

    vector_X: float = ligand_point[0] - center_point[0]
    vector_Y: float = ligand_point[1] - center_point[1]
    vector_Z: float = ligand_point[2] - center_point[2]
    vector_frac= [vector_X, vector_Y, vector_Z]

    near_point = [None]*3
    if(vector_frac[0] > 0.5):
        near_point[0] = ligand_point[0] - 1
    elif(vector_frac[0] < 0 and -vector_frac[0] > 0.5):
        near_point[0] = ligand_point[0] + 1
    else:
        near_point[0] = ligand_point[0]

    if (vector_frac[1] > 0.5):
        near_point[1] = ligand_point[1] - 1
    elif (vector_frac[1] < 0 and -vector_frac[1] > 0.5):
        near_point[1] = ligand_point[1] + 1
    else:
        near_point[1] = ligand_point[1]

    if (vector_frac[2] > 0.5):
        near_point[2] = ligand_point[2] - 1
    elif (vector_frac[2] < 0 and -vector_frac[2] > 0.5):
        near_point[2] = ligand_point[2] + 1
    else:
        near_point[2] = ligand_point[2]

    return near_point

'''
三维坐标中，分数坐标转换为笛卡尔坐标
'''
def Frac2Cart(point: list[float], basis_matrix: list[list[float]]):

    point = np.array(point)
    basis_matrix = np.array(basis_matrix)
    # 矩阵相乘
    result = np.dot(point, basis_matrix)

    if len(result) != 3 :
        raise Exception("输入的分数坐标或基矢坐标有误！")

    return result

'''
三维坐标中，笛卡尔坐标转换为分数坐标
'''
def Cart2Frac(point: list[float], basis_matrix: list[list[float]]):
    point = np.array(point)
    basis_matrix = np.array(basis_matrix)

    # 得到基矢矩阵的逆矩阵
    try:
        inverse_basis_matrix = np.linalg.inv(basis_matrix)
    except np.linalg.LinAlgError:
        print("矩阵不可逆")

    # 矩阵相乘
    result = np.dot(point, inverse_basis_matrix)

    if len(result) != 3 :
        raise Exception("输入的分数坐标或基矢坐标有误！")

    return result



# 测试
# basis_matrix = [[11.8347806931,0,0],[0,13.1122989655,0],[0.3300167970,0,12.6135519369]]
# point1 = [0.281680912,10.372487068,10.228734970]
# point2 = [10.369369507,10.071670532,10.047688484]
#
# point3 = [0.99429,0.18828,0.29964]
# point4 = [0.04377,0.33887,0.15855]
#
# near_point = get_near_periodic_point_v2(point3, point4)
# print(near_point)