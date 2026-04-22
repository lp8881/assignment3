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
<img width="1527" height="937" alt="2606009cff51b2963d2184d65c544838" src="https://github.com/user-attachments/assets/2d1677f9-57b0-4814-8398-a301ac206a9f" />






## Acknowledgement
