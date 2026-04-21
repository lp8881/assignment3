import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.transforms import euler_angles_to_matrix

class CameraOptimizer(nn.Module):
    def __init__(self, num_points, num_views, img_size=1024.0):
        super().__init__()
        # 1. 3D 点坐标 (N, 3) - 随机初始化，并将其推远一点以防止初始深度为负
        self.points_3d = nn.Parameter(torch.randn(num_points, 3) * 0.1)
        self.points_3d.data[:, 2] += 5.0 

        # 2. 相机外参: 旋转 (Euler角) 和 平移
        self.euler_angles = nn.Parameter(torch.zeros(num_views, 3))
        self.translations = nn.Parameter(torch.zeros(num_views, 3))
        
        # 3. 相机内参: 焦距 (初始化为一个合理的先验值)
        self.focal_length = nn.Parameter(torch.tensor([1000.0]))
        
        # 假设主点(Principal Point)在图像中心，不参与优化
        self.register_buffer("cx", torch.tensor(img_size / 2))
        self.register_buffer("cy", torch.tensor(img_size / 2))

    def forward(self):
        # 将 Euler 角转换为旋转矩阵 (V, 3, 3)
        R = euler_angles_to_matrix(self.euler_angles, convention="XYZ")
        T = self.translations  # (V, 3)

        # 批量矩阵乘法将 3D 点变换到相机坐标系
        # R: (V, 3, 3), points_3d.T: (3, N) -> (V, 3, N) -> transpose -> (V, N, 3)
        points_cam = torch.matmul(R, self.points_3d.T).transpose(1, 2) 
        points_cam = points_cam + T.unsqueeze(1) # 加上平移向量 (V, 1, 3)

        # 提取 Z 轴深度，添加 epsilon 避免除以 0
        z = points_cam[..., 2:3]
        z = torch.clamp(z, min=1e-4)

        # 针孔相机透视投影到 2D 像素平面
        # x_2d = f * (x / z) + cx
        p_2d_norm = points_cam[..., :2] / z
        
        # 乘上焦距并加上主点偏移
        proj_2d = p_2d_norm * self.focal_length
        proj_2d[..., 0] += self.cx
        proj_2d[..., 1] += self.cy

        return proj_2d

def save_colored_obj(filename, vertices, colors):
    """保存带有顶点颜色的 OBJ 文件 (如 MeshLab 支持的格式)"""
    # 确保颜色在 0-1 范围内
    if colors.max() > 1.0:
        colors = colors / 255.0
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    with open(filename, 'w') as f:
        for v, c in zip(vertices, colors):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
    print(f"[*] 3D 点云已保存至: {filename}")

def main():
    # ---------------- 1. 数据加载 ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用设备: {device}")

    # 模拟数据路径 (请确保在 data/ 目录下运行)
    points2d_path = 'data/points2d.npz'
    colors_path = 'data/points3d_colors.npy'
    
    data_2d = np.load(points2d_path)
    points_colors = np.load(colors_path)

    V = 50       # 视图数量
    N = 20000    # 点数量
    
    gt_2d_list = []
    mask_list = []

    for i in range(V):
        view_key = f"view_{i:03d}"
        pts = data_2d[view_key]  # shape: (20000, 3)
        
        # 假设前两维是 x, y，第三维是可见性/置信度(visibility mask)
        gt_2d_list.append(pts[:, :2])
        mask_list.append(pts[:, 2])

    gt_2d = torch.tensor(np.stack(gt_2d_list), dtype=torch.float32, device=device) # (V, N, 2)
    mask = torch.tensor(np.stack(mask_list), dtype=torch.float32, device=device)   # (V, N)

    # ---------------- 2. 模型与优化器设置 ----------------
    model = CameraOptimizer(num_points=N, num_views=V).to(device)
    
    # 使用 Adam 优化器
    optimizer = torch.optim.Adam([
        {'params': model.points_3d, 'lr': 1e-2},
        {'params': model.euler_angles, 'lr': 1e-3},
        {'params': model.translations, 'lr': 1e-3},
        {'params': model.focal_length, 'lr': 1e-2}
    ])

    # ---------------- 3. 优化循环 ----------------
    num_epochs = 1000
    loss_history = []

    print("[*] 开始优化...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # 前向投影
        pred_2d = model()
        
        # 计算 L2 重投影误差 (仅计算 mask > 0 的有效点)
        diff = pred_2d - gt_2d
        reprojection_error = torch.norm(diff, dim=-1) # (V, N)
        
        # 采用 Masked Mean 误差计算
        loss = (reprojection_error * mask).sum() / mask.sum().clamp(min=1.0)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:04d} | Loss (Reprojection Error): {loss.item():.4f} px | Focal Length: {model.focal_length.item():.2f}")

    # ---------------- 4. 可视化与保存 ----------------
    # 绘制 Loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Reprojection Error", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Pixels)")
    plt.title("Optimization Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig("optimization_loss.png")
    print("[*] Loss 曲线已保存至: optimization_loss.png")

    # 提取优化后的 3D 点保存为 OBJ
    final_points_3d = model.points_3d.detach().cpu().numpy()
    final_points_3d = -final_points_3d
    save_colored_obj("output_reconstruction.obj", final_points_3d, points_colors)

if __name__ == "__main__":
    main()