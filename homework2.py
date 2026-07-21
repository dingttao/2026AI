import os
# 禁用 Gradio 的后台数据统计，防止网络超时报错
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# ==========================================
# 1. 加载预训练的 ResNet50 模型
# ==========================================
print("正在加载预训练模型...")
# 使用官方推荐的最新权重加载方式
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.eval()  # 设置为评估模式

# 在 ImageNet 的 1000 个类别中，金鱼 (goldfish) 的索引是 1
GOLDFISH_CLASS_INDEX = 1

# ==========================================
# 2. 定义图像预处理流程
# ==========================================
# 必须使用与模型训练时相同的预处理参数
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
# 3. 定义预测函数
# ==========================================
def predict_goldfish_prob(image):
    if image is None:
        return "请先上传一张图片！"

    image = image.convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_batch = input_batch.to(device)

    with torch.no_grad():
        output = model(input_batch)

    # 【恢复原始概率】：不使用温度系数，保留模型最真实的判断
    probabilities = F.softmax(output[0], dim=0)
    goldfish_prob = probabilities[GOLDFISH_CLASS_INDEX].item() * 100

    # 【科学判定】：在 1000 分类中，概率大于 10% 即可判定为存在
    THRESHOLD = 10.0

    if goldfish_prob >= THRESHOLD:
        status = "✅ 判定结果：【存在】金鱼"
    else:
        status = "❌ 判定结果：【不存在】金鱼"

    # 格式化输出字符串
    result_str = f"{status}\n"
    result_str += "=" * 30 + "\n"
    result_str += f"原始检测概率为: {goldfish_prob:.2f}% (判定阈值: {THRESHOLD}%)\n\n"

    # 附加信息：看看模型到底识别成了什么
    top3_prob, top3_catid = torch.topk(probabilities, 3)
    categories = weights.meta["categories"]

    result_str += "模型认为最可能的前 3 个类别是：\n"
    for i in range(top3_prob.size(0)):
        cat_name = categories[top3_catid[i]]
        prob = top3_prob[i].item() * 100
        result_str += f"第 {i + 1} 名: {cat_name} ({prob:.2f}%)\n"

    return result_str


# ==========================================
# 4. 使用 Gradio 搭建网页 UI
# ==========================================
# 创建交互界面
interface = gr.Interface(
    fn=predict_goldfish_prob,  # 绑定的处理函数
    inputs=gr.Image(type="pil", label="请上传图片"),  # 输入组件：图像上传框
    outputs=gr.Textbox(label="检测结果", text_align="center"),  # 输出组件：文本框
    title="🐠 金鱼精准检测器 (二分类优化版)",
    description="上传任意一张图片，后台将输出该图片中是否存在金鱼。存在时概率将接近 100%，不存在时接近 0%。",
    theme="default"
)

# 启动网页服务
if __name__ == "__main__":
    print("启动网页服务中...")
    # share=False 默认在本地 7860 端口运行
    interface.launch(share=False)