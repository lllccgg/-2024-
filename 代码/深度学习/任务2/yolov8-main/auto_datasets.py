import os
import random
from glob import glob

def create_split_files():
    # --- 1. 配置路径 ---
    # 图片统一存放的文件夹
    img_dir = "E://yolov8-main//datasets//SHANZHI_new1//images"
    # dataset.yaml 文件所在的目录，用于生成 train.txt 和 val.txt
    dataset_yaml_dir = "E://yolov8-main//datasets//SHANZHI_new1"

    # --- 2. 获取所有图片 ---
    # 获取所有 .jpg 和 .png 图片的绝对路径
    image_paths = glob(os.path.join(img_dir, '*.jpg')) + glob(os.path.join(img_dir, '*.png'))
    
    # 检查是否找到了图片
    if not image_paths:
        print(f"错误：在目录 '{img_dir}' 中没有找到任何 .jpg 或 .png 图片。")
        return

    print(f"总共找到 {len(image_paths)} 张图片。")

    # --- 3. 随机打乱与分割 ---
    random.shuffle(image_paths)
    
    # 定义验证集比例
    val_split = 0.2
    split_index = int(len(image_paths) * (1 - val_split))
    
    train_paths = image_paths[:split_index]
    val_paths = image_paths[split_index:]

    # --- 4. 写入 .txt 文件 ---
    def write_to_txt(file_path, path_list):
        with open(file_path, 'w') as f:
            for path in path_list:
                f.write(path + '\n')
        print(f"成功生成文件: {file_path}，包含 {len(path_list)} 条记录。")

    # 构建 train.txt 和 val.txt 的完整路径
    train_txt_path = os.path.join(dataset_yaml_dir, 'train.txt')
    val_txt_path = os.path.join(dataset_yaml_dir, 'val.txt')

    # 写入文件
    write_to_txt(train_txt_path, train_paths)
    write_to_txt(val_txt_path, val_paths)
    
    print("\n数据集索引文件生成完毕！")

if __name__ == "__main__":
    create_split_files()