## 6D Pose Estimation Proposal 


### FastSAM Segementation

Uses the FastSAM-x Model for image segmentation.
Code from FastSAM github [repo](https://github.com/CASIA-IVA-Lab/FastSAM) for inference.

<img src="https://github.com/user-attachments/assets/76b833f8-8cca-4325-ac09-78c6b8437ea0" width="200">
<img src="https://github.com/user-attachments/assets/423d97a3-7fe4-43e8-bc30-3d68b2e64c90" width="200">
<img src="https://github.com/user-attachments/assets/0d95384e-0700-4960-a679-6895d43526ff" width="200">
<img src="https://github.com/user-attachments/assets/4d0aeb39-ce7a-45bb-a4ba-2c2c40ff0d73" width="200">

---

### Depth Image And Point Cloud Generation From RGB Image

**Pipeline**

Using the [LiheYoung/depth-anything-Large-hf](https://huggingface.co/LiheYoung/depth-anything-large-hf) or [intel-isl/MiDaS](https://pytorch.org/hub/intelisl_midas_v2/) models a depth image is generated from RGB image.

Then Open3d Library is used to filter and create the point cloud.

Open3d is also used for visualization.


<img src="https://github.com/pacificblaster1708/6D-Pose-Estimation-Proposal/blob/main/Point%20Cloud%20and%20RGB-Depth/0.png" width="500">

https://github.com/user-attachments/assets/3df49a1d-a72a-4253-885d-5a942bdcd92e

<img src="https://github.com/pacificblaster1708/6D-Pose-Estimation-Proposal/blob/main/Point%20Cloud%20and%20RGB-Depth/0_depth.png" width="500">


<img src="https://github.com/pacificblaster1708/6D-Pose-Estimation-Proposal/blob/main/Point%20Cloud%20and%20RGB-Depth/Actual_image_open3d.jpg" width="250"><img src="https://github.com/pacificblaster1708/6D-Pose-Estimation-Proposal/blob/main/Point%20Cloud%20and%20RGB-Depth/rgb_depth_open3d.jpg" width="250">

Point Cloud

<img src="https://github.com/pacificblaster1708/6D-Pose-Estimation-Proposal/blob/main/Point%20Cloud%20and%20RGB-Depth/Pointcloud_open3d.jpg" width="250">
