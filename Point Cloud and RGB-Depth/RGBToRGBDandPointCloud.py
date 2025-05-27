import argparse
import numpy as np
import open3d as o3d

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path", type=str, default="./Images/Controllers.jpg", help="image Path"
    )
    parser.add_argument(
        "--results_path", type=str, default="./Results/", help="results Folder"
    )
    parser.add_argument(
        "--models_path", type=str, default="./Models/", help="models Folder"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="cuda:[0,1,2,3,4] or cpu"
    )

    return parser.parse_args()

def depthToPointCloudOrthographic(depthMap, image, scaleFactor = 255):
    height,width = depthMap.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    z = (depthMap/scaleFactor) * height / 2


    points = np.stack((x,y,z), axis = -1).reshape(-1,3)

    mask = points[:,2] != 0
    points = points[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

    colors = image.reshape(-1, 3)[mask] / 255.0



    pcd.colors = o3d.utility.Vector3dVector(colors)

    _, ind = pcd.remove_statistical_outlier(nb_neighbors = 15,std_ratio=1)

    #inlier_cloud = pcd.select_by_index(ind)
    inlier_cloud = pcd
    return inlier_cloud,z ,height, width

def main(args):

    print("Loading Modules......")
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    import cv2

    import open3d as o3d

    import torch

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    print("Modules Loaded.")

    if(args.device != None):
        DEVICE = args.device
    else:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    imagePath = Path(args.image_path)

    imageName = imagePath.name
    imageName = imageName.split('.')[0]

    numImages = 1

    selectedImages = []    
    imagePath = str(imagePath)
    selectedImage = cv2.imread(imagePath)
    selectedImage = cv2.cvtColor(selectedImage,cv2.COLOR_BGR2RGB)
    selectedImages.append(selectedImage)

    models = ["LiheYoung/depth-anything-Large-hf"]

    model = models[0]

    processor = AutoImageProcessor.from_pretrained(model,use_fast=True)

    model = AutoModelForDepthEstimation.from_pretrained(model).to(DEVICE)

    depthImages = []

    for i in range(numImages):
        depthImageInput = processor(images=selectedImages[i],return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            inferenceOutputs = model(**depthImageInput)
            outputDepth = inferenceOutputs.predicted_depth

        outputDepth = outputDepth.squeeze().cpu().numpy()

        depthImages.append([selectedImages[i],outputDepth])

    plt.rcParams["figure.dpi"] = 300

    for i in range(numImages):
        fig, axs = plt.subplots(2,1)

        axs[0].imshow(depthImages[i][0])
        axs[0].set_title("Depth Estimation")
        axs[1].imshow(depthImages[i][1])

        plt.show()    

    for i in range(numImages):
        depthImage = depthImages[i][1]
        colorImage = depthImages[i][0]

        width, height = depthImage.shape

        depthImage = (depthImage * 255 / np.max(depthImage)).astype('uint8')
        colorImage = cv2.resize(colorImage,(height,width))

        cv2.imwrite(args.results_path + imageName + '.png', cv2.cvtColor(colorImage, cv2.COLOR_BGR2RGB))
        cv2.imwrite(args.results_path + imageName + '_depth.png', depthImage)

    i = 0

    depthImage = depthImages[i][1]
    colorImage = depthImages[i][0]
    width, height = depthImage.shape

    depthImage = (depthImage * 255 / np.max(depthImage)).astype('uint8')
    colorImage = cv2.resize(colorImage, (height, width))

    depthO3d = o3d.geometry.Image(depthImage)
    imageO3d = o3d.geometry.Image(colorImage)

    rgbdImage = o3d.geometry.RGBDImage.create_from_color_and_depth(imageO3d,depthO3d,convert_rgb_to_intensity = False)

    cameraIntrinsic = o3d.camera.PinholeCameraIntrinsic()

    fx = fy = width * 0.8
    cx, cy = width / 2, height / 2


    cameraIntrinsic.set_intrinsics(width, height,fx,fy,cx,cy)

    pcdRaw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbdImage, cameraIntrinsic)

    cl , ind = pcdRaw.remove_statistical_outlier(nb_neighbors=20, std_ratio=6.0)
    pcd = pcdRaw.select_by_index(ind)

    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()
    

    exporting = True

    i = 0

    depthMap = cv2.imread(args.results_path + imageName + '_depth.png', cv2.IMREAD_ANYDEPTH)
    image = cv2.imread(args.results_path + imageName + '.png', cv2.IMREAD_ANYCOLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



    pointCloud, z, height, width = depthToPointCloudOrthographic(depthMap,image)

    o3d.visualization.draw_geometries([pointCloud])

    if exporting:
        o3d.io.write_point_cloud(args.models_path + imageName + ".ply", pointCloud)


if __name__ == "__main__":
    args = parseArgs()
    main(args)