import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os

# ==========================================
# 1. 加载预训练的 ResNet50 模型
# ==========================================
print("正在加载预训练模型，请稍候...")
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.eval()  # 设置为评估模式

# 获取官方的 1000 个类别名称标签
categories = weights.meta["categories"]
# 在 ImageNet 的 1000 个类别中，金鱼 (goldfish) 的索引是 1
GOLDFISH_CLASS_INDEX = 1

# ==========================================
# 2. 定义图像预处理流程
# ==========================================
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ==========================================
# 3. 定义核心预测函数
# ==========================================
def predict_image(image_path):
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"❌ 错误：找不到图片文件，请检查路径是否正确！({image_path})")
        return

    try:
        # 读取图片并强制转换为 RGB
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ 错误：无法读取图片，请确保它是一张有效的图像文件。({e})")
        return

    # 预处理图像
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # 如果有 GPU 则使用 GPU 加速
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_batch = input_batch.to(device)

    # 模型推理
    with torch.no_grad():
        output = model(input_batch)

    # 计算概率
    probabilities = F.softmax(output[0], dim=0)
    goldfish_prob = probabilities[GOLDFISH_CLASS_INDEX].item()

    # 获取排名前 3 的预测结果
    top3_prob, top3_catid = torch.topk(probabilities, 3)

    # 打印结果
    print("\n" + "=" * 45)
    print(f"📸 分析图片: {os.path.basename(image_path)}")
    print(f"🐟 【金鱼 (goldfish)】的概率为: {goldfish_prob * 100:.2f}%")
    print("-" * 45)
    print("🏆 模型认为最可能的前 3 个类别是：")

    for i in range(top3_prob.size(0)):
        cat_name = categories[top3_catid[i]]
        prob = top3_prob[i].item() * 100
        print(f"   第 {i + 1} 名: {cat_name} ({prob:.2f}%)")
    print("=" * 45 + "\n")


# ==========================================
# 4. 交互式主程序
# ==========================================
if __name__ == "__main__":
    print("\n✅ 模型加载完毕！")

    while True:
        # 接收用户输入的路径
        user_input = input("👉 请输入图片路径 (输入 q 退出程序): ")

        if user_input.strip().lower() == 'q':
            print("👋 程序已退出。")
            break

        if not user_input.strip():
            continue

        # 清理路径字符串（自动去除用户复制路径时可能带上的双引号）
        clean_path = user_input.strip().strip('"').strip("'")

        # 执行预测
        predict_image(clean_path)