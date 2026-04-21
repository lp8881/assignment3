# asignment3
## Task 1: Implement Bundle Adjustment with PyTorch

## Requirements

To install requirements:

```setup
python -m pip install -r requirements.txt
```


## Running

Reconstructe 3D point cloud, run:

```basic
python Bundle_Adjustment.py
```

## Results (need add more result images)
### Loss：
<img width="800" height="500" alt="optimization_loss" src="https://github.com/user-attachments/assets/f4e70f34-ea8b-4354-b8d7-04d02406d14a" />
### Reconstructed 3D point cloud:
Use MeshLab to open output_reconstruction.obj
<img width="1920" height="1200" alt="output_reconstruction_meshlab" src="https://github.com/user-attachments/assets/9ac3fefc-218d-4104-8142-7307e739b7fb" />

## Task 2: 3D Reconstruction with COLMAP

使用 [COLMAP](https://colmap.github.io/) 命令行工具，对 `data/images/` 中的 50 张渲染图像进行完整的三维重建。

### 具体步骤：

1. **特征提取** (Feature Extraction)(电脑无GPU要将gpu_index设置为0)
2. **特征匹配** (Feature Matching)(电脑无GPU要将gpu_index设置为0)
3. **稀疏重建** (Sparse Reconstruction / Mapper) — 即 COLMAP 内部的 Bundle Adjustment
4. **稠密重建** (Dense Reconstruction) — 包括 Image Undistortion、Patch Match Stereo、Stereo Fusion
5. **结果展示** — 在报告中展示稀疏点云或稠密点云的截图（可使用 [MeshLab](https://www.meshlab.net/) 查看 `.ply` 文件）

完整的命令行脚本见 [run_colmap.sh](run_colmap.sh)，可参考 [COLMAP CLI Tutorial](https://colmap.github.io/cli.html) 了解各步骤详情。

#### COLMAP 安装：
- **Linux**：参考 [官方安装文档](https://colmap.github.io/install.html) 从源码编译（需开启 CUDA 支持），或使用 `conda install -c conda-forge colmap`
- **Windows**：从 [COLMAP Releases](https://github.com/colmap/colmap/releases) 下载 `COLMAP-dev-windows-cuda.zip`，解压后将目录加入 PATH 即可使用

稠密重建需要 CUDA GPU；如无 GPU，可只完成到稀疏重建步骤。


## Acknowledgement
