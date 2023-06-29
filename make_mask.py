import os
import cv2
import numpy as np


def shrink_quadrilateral(box, shrink_factor):
    """
    对四边形框进行向内收缩
    
    参数：
    box: numpy数组，表示四边形框的顶点坐标，维度为(4, 2)
    shrink_factor: 收缩因子，是一个0到1之间的值
    
    返回值：
    shrinked_box: numpy数组，表示收缩后的四边形框的顶点坐标，维度为(4, 2)
    """
    # 计算四边形的中心点
    center = np.mean(box, axis=0)
    
    # 计算每个顶点与中心点的向量
    vectors = box - center
    
    # 根据收缩因子计算新的顶点坐标
    shrinked_vectors = vectors * (1 - shrink_factor)
    
    # 计算收缩后的四边形框的顶点坐标
    shrinked_box = np.round(center + shrinked_vectors).astype(int)
    
    return shrinked_box

# 设置原始图片文件夹和标注文件夹的路径
image_folder = "train_image"
gt_folder = "train_gt"

# 设置保存二值分割图的文件夹路径
output_folder = "segmentation"

# 确保保存二值分割图的文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历所有图片文件
for filename in os.listdir(image_folder):
    # 读取原始图片
    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)
    
    # 读取对应的标注文件
    gt_filename ="gt_"+ os.path.splitext(filename)[0] + ".txt"
    gt_path = os.path.join(gt_folder, gt_filename)
    
    # 创建一个全黑的二值分割图像
    height, width, _ = image.shape
    segmentation = np.zeros((height, width), dtype=np.uint8)
    
    # 逐行读取标注文件并解析顶点坐标
    with open(gt_path, 'r') as f:
    	
        for line in f:
            line=line.strip('\ufeff')
            # 解析顶点坐标
            values=line.strip().split(",")
        
            points = list(map(int,values[:8]))
            vertices = np.array(points, dtype=np.int32).reshape((-1, 2))
            vertices=shrink_quadrilateral(vertices,0.1)
            
            # 创建对应区域的掩膜
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [vertices], color=255)
            
            # 将该区域的掩膜添加到二值分割图像中
            segmentation = cv2.bitwise_or(segmentation, mask)
    
    # 保存二值分割图像
    output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
    cv2.imwrite(output_path, segmentation)
    
    print(f"Created segmentation image for {filename}")
