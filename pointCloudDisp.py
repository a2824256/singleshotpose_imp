import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 点云文件路径
file_path = 'LINEMOD/milk/milk.ply'
# 打开文件
f = open(file_path, 'r')
# 读取点云数据
content = f.read()
# 关闭文件
f.close()
# 字符替换
lines = content.replace('\n', ',')
# 字符分割
lines = lines.split(',')
# 获取配置信息结束标识符end_header的行索引
config_final_index = lines.index("end_header") + 1
# 删除前面的配置信息
del lines[0:config_final_index]
lines.pop()
# 转numpy数组
x = []
y = []
z = []
np_points = np.array(lines)
for i in range(0, len(lines)):
    info_groups = lines[i].split(' ')
    x.append(info_groups[0])
    y.append(info_groups[1])
    z.append(info_groups[2])
container = [x, y, z]
np_points = np.array(container).astype(float)

min_x = np.min(np_points[0,:])
max_x = np.max(np_points[0,:])
min_y = np.min(np_points[1,:])
max_y = np.max(np_points[1,:])
min_z = np.min(np_points[2,:])
max_z = np.max(np_points[2,:])
corners = np.array([[min_x, min_y, min_z],
                    [min_x, min_y, max_z],
                    [min_x, max_y, min_z],
                    [min_x, max_y, max_z],
                    [max_x, min_y, min_z],
                    [max_x, min_y, max_z],
                    [max_x, max_y, min_z],
                    [max_x, max_y, max_z]])
print(corners)
fig = plt.figure()
ax = plt.subplot(111, projection='3d')
# 遍历数组
for item in corners:
    # info_groups = item.split(' ')
    ax.scatter(float(item[0]), float(item[1]), float(item[2]), c='r')
ax.set_zlabel('Z')
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()