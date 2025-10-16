import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from PIL import Image
import random

# 数据集路径和参数配置
datasets_path = './datasets'
normalized_size = (32, 32)
pretrained_model_path = './lenet5_armor_finetuned.pth'  # 预训练模型路径
finetuned_model_path = './lenet5_armor_end.pth'  # 微调后模型保存路径
batch_size = 32
num_epochs = 10  # 微调使用较少的epoch
learning_rate = 0.0001  # 微调使用较小的学习率

# 图像预处理函数（与原始训练保持一致）
def process_img(img):
    if (len(img.shape) > 2):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    resized = cv2.resize(gray, normalized_size)
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary

# 自定义光干扰噪声变换类
class LightInterferenceTransform:
    """模拟装甲板识别中的条灯光干扰"""
    
    def __init__(self, noise_prob=0.7):
        self.noise_prob = noise_prob
    
    def __call__(self, img):
        if random.random() > self.noise_prob:
            return img
        
        # 转换为numpy数组进行处理
        if isinstance(img, Image.Image):
            img_array = np.array(img)
        else:
            img_array = img
        
        # 随机选择一种或多种噪声类型
        noise_types = []
        if random.random() < 0.4:  # 40%概率添加高斯噪声
            noise_types.append('gaussian')
        if random.random() < 0.3:  # 30%概率添加椒盐噪声
            noise_types.append('salt_pepper')
        if random.random() < 0.5:  # 50%概率添加光斑干扰
            noise_types.append('light_spots')
        if random.random() < 0.3:  # 30%概率添加条纹干扰
            noise_types.append('light_stripes')
        
        # 如果没有选中任何噪声，至少添加一种
        if not noise_types:
            noise_types = [random.choice(['gaussian', 'light_spots'])]
        
        # 应用选中的噪声
        for noise_type in noise_types:
            if noise_type == 'gaussian':
                img_array = self._add_gaussian_noise(img_array)
            elif noise_type == 'salt_pepper':
                img_array = self._add_salt_pepper_noise(img_array)
            elif noise_type == 'light_spots':
                img_array = self._add_light_spots(img_array)
            elif noise_type == 'light_stripes':
                img_array = self._add_light_stripes(img_array)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _add_gaussian_noise(self, img):
        """添加高斯噪声模拟电子噪声"""
        noise_std = random.uniform(5, 20)  # 噪声标准差
        noise = np.random.normal(0, noise_std, img.shape)
        noisy_img = img.astype(np.float32) + noise
        return np.clip(noisy_img, 0, 255)
    
    def _add_salt_pepper_noise(self, img):
        """添加椒盐噪声模拟像素损坏"""
        noise_ratio = random.uniform(0.01, 0.05)  # 噪声比例1%-5%
        noisy_img = img.copy()
        
        # 盐噪声（白点）
        salt_coords = np.random.random(img.shape) < noise_ratio / 2
        noisy_img[salt_coords] = 255
        
        # 椒噪声（黑点）
        pepper_coords = np.random.random(img.shape) < noise_ratio / 2
        noisy_img[pepper_coords] = 0
        
        return noisy_img
    
    def _add_light_spots(self, img):
        """添加光斑干扰模拟条灯反射"""
        noisy_img = img.copy().astype(np.float32)
        h, w = img.shape[:2]
        
        # 随机生成1-3个光斑
        num_spots = random.randint(1, 3)
        
        for _ in range(num_spots):
            # 光斑中心位置
            center_x = random.randint(0, w-1)
            center_y = random.randint(0, h-1)
            
            # 光斑半径和强度
            radius = random.uniform(2, 8)
            intensity = random.uniform(50, 150)
            
            # 创建光斑掩码
            y, x = np.ogrid[:h, :w]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            # 添加高斯衰减的光斑
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            gaussian_mask = np.exp(-(distance**2) / (2 * (radius/2)**2))
            
            # 应用光斑
            noisy_img[mask] += intensity * gaussian_mask[mask]
        
        return np.clip(noisy_img, 0, 255)
    
    def _add_light_stripes(self, img):
        """添加条纹干扰模拟LED条灯"""
        noisy_img = img.copy().astype(np.float32)
        h, w = img.shape[:2]
        
        # 随机选择水平或垂直条纹
        if random.random() < 0.5:
            # 水平条纹
            stripe_width = random.randint(1, 3)
            stripe_spacing = random.randint(4, 8)
            intensity = random.uniform(30, 80)
            
            for y in range(0, h, stripe_spacing):
                end_y = min(y + stripe_width, h)
                noisy_img[y:end_y, :] += intensity
        else:
            # 垂直条纹
            stripe_width = random.randint(1, 3)
            stripe_spacing = random.randint(4, 8)
            intensity = random.uniform(30, 80)
            
            for x in range(0, w, stripe_spacing):
                end_x = min(x + stripe_width, w)
                noisy_img[:, x:end_x] += intensity
        
        return np.clip(noisy_img, 0, 255)

# 自定义数据集类
class ArmorDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx].astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.LongTensor([self.labels[idx]])[0]
        return image, label

# 数据加载函数（与原始保持一致）
def load_datasets(path):
    train_datas = []
    labels = []
    print("开始加载数据集......")
    
    class_counts = {}
    
    for foldername in os.listdir(path):
        folder_path = os.path.join(path, foldername)
        if os.path.isdir(folder_path):
            try:
                label = int(foldername)
                if label < 0 or label > 8:
                    print(f"跳过超出范围的文件夹：{foldername}")
                    continue
                    
                print(f"正在加载文件夹 {foldername} (标签: {label})...")
                
                image_count = 0
                for filename in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, filename)
                    
                    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        continue
                        
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        processed_img = process_img(img)
                        train_datas.append(processed_img)
                        labels.append(label)
                        image_count += 1
                    else:
                        print(f"无法读取图像：{img_path}")
                
                class_counts[label] = image_count
                print(f"文件夹 {foldername} 加载完成，共 {image_count} 张图像")
                
            except ValueError:
                print(f"跳过非数字文件夹：{foldername}")
                continue
    
    print("\n=== 数据集加载统计 ===")
    for label in sorted(class_counts.keys()):
        print(f"类别 {label}: {class_counts[label]} 张图像")
    print(f"总计: {len(train_datas)} 张图像")
    
    return np.array(train_datas), np.array(labels)

# LeNet-5网络结构（与原始保持一致）
class LeNet5_Armor(nn.Module):
    def __init__(self, num_classes=9):
        super(LeNet5_Armor, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(16, 120, kernel_size=6)
        
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = torch.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 计算准确率函数
def compute_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 微调主函数
def finetune_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载数据集
    print("加载数据集...")
    images, labels = load_datasets(datasets_path)
    
    # 2. 数据集分割
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 3. 定义增强的数据变换（关键改进 - 添加光干扰噪声）
    # 训练集使用更强的数据增强，专门针对比例变化和光干扰
    train_transform = transforms.Compose([
        # 光干扰噪声增强（新增）- 模拟条灯干扰
        LightInterferenceTransform(noise_prob=0.8),  # 80%概率添加光干扰
        
        # 随机仿射变换：旋转、平移、缩放
        transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.8, 1.3)),
        # 随机透视变换：模拟不同角度观察
        transforms.RandomPerspective(distortion_scale=0.2, p=0.6),
        # 随机裁剪和缩放：模拟不同比例的ROI
        transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(0.7, 1.4)),
        # 随机水平翻转（如果装甲板数字允许）
        transforms.RandomHorizontalFlip(p=0.3),
        # 随机垂直翻转（谨慎使用）
        transforms.RandomVerticalFlip(p=0.1),
        # 颜色抖动（对二值图像效果有限，但可以增加鲁棒性）
        transforms.ColorJitter(brightness=0.3, contrast=0.3),  # 增强亮度对比度变化
        transforms.ToTensor()
    ])

    # 测试集只进行基本转换
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # 4. 创建数据集和数据加载器
    train_dataset = ArmorDataset(X_train, y_train, transform=train_transform)
    test_dataset = ArmorDataset(X_test, y_test, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 5. 加载预训练模型
    print("加载预训练模型...")
    model = LeNet5_Armor(num_classes=9).to(device)
    
    # 检查预训练模型是否存在
    if os.path.exists(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载预训练模型: {pretrained_model_path}")
        
        # 可选：显示预训练模型的性能
        if 'test_accuracies' in checkpoint:
            best_pretrained_acc = max(checkpoint['test_accuracies'])
            print(f"预训练模型最佳测试准确率: {best_pretrained_acc:.2f}%")
    else:
        print(f"警告：预训练模型 {pretrained_model_path} 不存在，将从头开始训练")
    
    # 6. 设置优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    # 微调使用较小的学习率
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters())}")
    print("已启用光干扰噪声增强，模拟条灯干扰场景")
    
    # 7. 微调训练循环
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # 记录初始性能
    initial_train_acc = compute_accuracy(model, train_loader, device)
    initial_test_acc = compute_accuracy(model, test_loader, device)
    print(f"微调前 - 训练准确率: {initial_train_acc:.2f}%, 测试准确率: {initial_test_acc:.2f}%")
    
    print("开始微调训练...")
    best_test_acc = initial_test_acc
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # 计算准确率
        train_acc = compute_accuracy(model, train_loader, device)
        test_acc = compute_accuracy(model, test_loader, device)
        avg_loss = total_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'平均损失: {avg_loss:.4f}')
        print(f'训练准确率: {train_acc:.2f}%')
        print(f'测试准确率: {test_acc:.2f}%')
        
        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f'新的最佳测试准确率: {best_test_acc:.2f}%')
        
        print('-' * 50)
        scheduler.step()
    
    # 8. 最终评估
    model.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    # 打印详细评估结果
    print("\n=== 微调后最终评估结果 ===")
    final_acc = accuracy_score(y_true, y_pred) * 100
    print(f"微调前测试准确率: {initial_test_acc:.2f}%")
    print(f"微调后测试准确率: {final_acc:.2f}%")
    print(f"准确率提升: {final_acc - initial_test_acc:.2f}%")
    print("✨ 已应用光干扰噪声增强，提升模型对条灯干扰的鲁棒性")
    print("\n分类报告:")
    print(classification_report(y_true, y_pred))
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('微调后混淆矩阵（含光干扰增强）')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()
    
    # 9. 保存微调后的模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'initial_test_acc': initial_test_acc,
        'final_test_acc': final_acc,
        'improvement': final_acc - initial_test_acc,
        'light_interference_enabled': True  # 标记使用了光干扰增强
    }, finetuned_model_path)
    
    print(f"微调后模型已保存至: {finetuned_model_path}")
    
    # 绘制微调训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('微调训练损失（含光干扰增强）')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='训练准确率')
    plt.plot(test_accuracies, label='测试准确率')
    plt.axhline(y=initial_test_acc, color='r', linestyle='--', label=f'微调前: {initial_test_acc:.1f}%')
    plt.title('微调准确率变化（含光干扰增强）')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    improvement_per_epoch = [acc - initial_test_acc for acc in test_accuracies]
    plt.plot(improvement_per_epoch, 'g-', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title('准确率提升趋势')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Improvement (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 可视化光干扰效果的辅助函数
def visualize_light_interference_effects():
    """可视化光干扰增强效果"""
    print("=== 光干扰增强效果可视化 ===")
    
    # 创建一个示例图像
    sample_img = np.ones((32, 32), dtype=np.uint8) * 128
    # 添加一些数字特征
    cv2.rectangle(sample_img, (8, 8), (24, 24), 255, -1)
    cv2.rectangle(sample_img, (12, 12), (20, 20), 0, -1)
    
    # 创建光干扰变换
    light_transform = LightInterferenceTransform(noise_prob=1.0)
    
    # 生成多个增强样本
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes[0, 0].imshow(sample_img, cmap='gray')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    for i in range(7):
        row = i // 4
        col = (i + 1) % 4
        
        # 应用光干扰
        pil_img = Image.fromarray(sample_img)
        augmented = light_transform(pil_img)
        augmented_array = np.array(augmented)
        
        axes[row, col].imshow(augmented_array, cmap='gray')
        axes[row, col].set_title(f'光干扰样本 {i+1}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.suptitle('光干扰增强效果展示', y=1.02)
    plt.show()

if __name__ == "__main__":
    # 可选：先可视化光干扰效果
    print("是否要先查看光干扰增强效果？(y/n): ", end="")
    choice = input().lower()
    if choice == 'y':
        visualize_light_interference_effects()
    
    finetune_model()