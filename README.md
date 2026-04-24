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
<img width="800" height="500" alt="optimization_loss" src="https://github.com/lp8881/assignment3/blob/main/assignment3/optimization_loss.png" />


### Reconstructed 3D point cloud:

Use MeshLab to open output_reconstruction.obj
<img width="800" height="500" alt="MeshLab1" src="https://github.com/lp8881/assignment3/blob/main/assignment3/result_pic/MeshLab1.png" />
<img width="800" height="500" alt="MeshLab1" src="https://github.com/lp8881/assignment3/blob/main/assignment3/result_pic/MeshLab2.png" />
<img width="800" height="500" alt="MeshLab1" src="https://github.com/lp8881/assignment3/blob/main/assignment3/result_pic/MeshLab3.png" />

## Task 2: 3D Reconstruction with COLMAP

使用 [COLMAP](https://colmap.github.io/) 命令行工具，对 `data/images/` 中的 50 张渲染图像进行完整的三维重建。
<img width="1527" height="937" alt="colmap" src="<img width="800" height="500" alt="MeshLab1" src="https://github.com/lp8881/assignment3/blob/main/assignment3/result_pic/colmap.png" />" />






## Acknowledgement
