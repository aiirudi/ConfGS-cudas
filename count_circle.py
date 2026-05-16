from plyfile import PlyData

def count_spheres_in_ply(ply_file):
    # 读取 .ply 文件
    ply_data = PlyData.read(ply_file)
    
    # 获取顶点数据
    vertices = ply_data['vertex'].data
    
    # 返回顶点的数量（即小球的数量）
    return len(vertices)

# 示例：统计 .ply 文件中的小球数量
ply_file = '/workspace/dataset/mip_nerf360/playroom/sparse/0/points3D.ply'
num_spheres = count_spheres_in_ply(ply_file)
print(f"Number of spheres: {num_spheres}")
