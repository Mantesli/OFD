import os
import cv2
import numpy as np
import base64
import json
import time
from pathlib import Path
from openai import OpenAI

# =================配置区域=================
# 请确保环境变量 DASHSCOPE_API_KEY 已设置，或者直接在下面填入字符串
API_KEY = "sk-ef7db77064064747936dd65767cbd794"
# 测试图片路径 
TEST_IMAGE_PATH = r"E:\work\oilfield-leak-detection-v4\data\original\noleak001_007408.jpg"
OUTPUT_PATH = r"E:\work\oilfield-leak-detection-v4\results\final_workflow_result.jpg"
# =========================================

# 初始化 DashScope 客户端
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def encode_image_to_base64(cv2_img):
    """将 OpenCV 图片转换为 Base64 格式"""
    _, buffer = cv2.imencode('.jpg', cv2_img)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"


# ================= Stage 1: 端侧海选 (Local Proposal) =================
# 复用之前的多尺度+融合逻辑
def get_candidates_locally(image_path):
    print(f"🚀 [Stage 1] 正在进行本地多尺度海选: {image_path}")
    full_img = cv2.imread(str(image_path))
    if full_img is None: raise FileNotFoundError(f"找不到图片: {image_path}")

    h, w = full_img.shape[:2]
    mid = w // 2

    # 切割图像 (假设左红外，右可见光)
    ir_part = full_img[:, :mid]
    rgb_part = full_img[:, mid:]
    ir_gray = cv2.cvtColor(ir_part, cv2.COLOR_BGR2GRAY)

    # 多尺度检测
    scales = [1.0, 0.5, 0.25]
    all_boxes = []

    for s in scales:
        # 缩放
        width = int(ir_gray.shape[1] * s)
        height = int(ir_gray.shape[0] * s)
        ir_resized = cv2.resize(ir_gray, (width, height))
        rgb_resized = cv2.resize(rgb_part, (width, height))

        # 红外流
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        ir_blur = cv2.GaussianBlur(clahe.apply(ir_resized), (5, 5), 0)
        _, ir_mask = cv2.threshold(ir_blur, 200, 255, cv2.THRESH_BINARY)

        # RGB流
        gray = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2GRAY)
        _, rgb_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        # 融合
        if ir_mask.shape != rgb_mask.shape:
            rgb_mask = cv2.resize(rgb_mask, (ir_mask.shape[1], ir_mask.shape[0]))
        final_mask = cv2.bitwise_and(ir_mask, rgb_mask)

        # 提取框并还原坐标
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > (50 * s * s):
                x, y, w_box, h_box = cv2.boundingRect(c)
                all_boxes.append([int(x / s), int(y / s), int(w_box / s), int(h_box / s)])

    # 掩膜融合 (去碎片化)
    if not all_boxes: return [], rgb_part, ir_part

    canvas = np.zeros(ir_gray.shape, dtype=np.uint8)
    for (x, y, w_box, h_box) in all_boxes:
        cv2.rectangle(canvas, (x, y), (x + w_box, y + h_box), 255, -1)

    # 使用 20x20 核进行聚合
    kernel = np.ones((20, 20), np.uint8)
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_candidates = []
    for c in contours:
        x, y, w_box, h_box = cv2.boundingRect(c)
        if w_box * h_box > 500:  # 忽略微小噪点
            final_candidates.append([x, y, w_box, h_box])

    print(f"✅ [Stage 1] 海选完成，发现 {len(final_candidates)} 个疑似目标。")
    return final_candidates, rgb_part, ir_part


# ================= Stage 2: 云侧决断 (Cloud Reasoning) =================
def verify_with_qwen(crop_rgb, crop_ir):
    # 拼图：将红外图(转为BGR)和RGB图横向拼接
    if len(crop_ir.shape) == 2:
        crop_ir = cv2.cvtColor(crop_ir, cv2.COLOR_GRAY2BGR)

    # 为了让模型看得更清楚，如果是极小的图，放大一点
    if crop_rgb.shape[0] < 64:
        crop_rgb = cv2.resize(crop_rgb, (0, 0), fx=2, fy=2)
        crop_ir = cv2.resize(crop_ir, (0, 0), fx=2, fy=2)

    combined_img = np.hstack((crop_ir, crop_rgb))
    base64_img = encode_image_to_base64(combined_img)

    prompt = """
    你是一个油田工业视觉专家。图片左侧是红外热像（高亮区代表高温），右侧是可见光。
    请判断图中是否包含【地面石油泄漏】。

    必须严格区分：
    1. 【泄漏 (Positive)】：
       - 形状不规则、边缘呈锯齿状或羽化状。
       - 看起来像液体渗透或扩散。
       - 红外有温差，且可见光下为黑色。
    2. 【干扰 (Negative)】：
       - 管道/设备：笔直的线条、规则的几何长条。
       - 阴影：边缘锐利、几何形状规则。
       - 车辆/石头：孤立的固体形态。

    请仅输出 JSON：
    {"is_leak": true/false, "confidence": "high/medium/low", "reason": "简述判断依据"}
    """

    try:
        completion = client.chat.completions.create(
            model="qwen2-vl-72b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": base64_img}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=0.01,  # 降低随机性
        )
        content = completion.choices[0].message.content
        # 简单的 JSON 清洗（防止模型输出 markdown 符号）
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"⚠️ API 调用失败: {e}")
        return {"is_leak": False, "reason": "API Error"}


# ================= 主流程 =================
def main():
    if not API_KEY:
        print("❌ 错误: 未设置 API Key")
        return

    # 1. 本地海选
    candidates, full_rgb, full_ir = get_candidates_locally(TEST_IMAGE_PATH)

    if not candidates:
        print("未发现任何疑似目标，流程结束。")
        return

    # 2. 循环送审
    result_img = full_rgb.copy()
    print(f"\n🚀 [Stage 2] 开始云端 AI 复核 (Qwen2-VL-72B)...")

    for i, (x, y, w, h) in enumerate(candidates):
        print(f"\n--- 处理候选区 {i + 1}/{len(candidates)} ---")

        # 裁剪 (扩边 10 像素以保留上下文)
        pad = 10
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(full_rgb.shape[1], x + w + pad), min(full_rgb.shape[0], y + h + pad)

        crop_rgb = full_rgb[y1:y2, x1:x2]
        crop_ir = full_ir[y1:y2, x1:x2]

        # 调用 API
        start_time = time.time()
        result = verify_with_qwen(crop_rgb, crop_ir)
        cost_time = time.time() - start_time

        print(f"⏱️ 耗时: {cost_time:.2f}s")
        print(f"🤖 结论: {result}")

        # 绘制结果
        if result.get("is_leak", False):
            # 确诊泄漏：画红框 + 粗体字
            color = (0, 0, 255)
            label = f"LEAK ({result.get('confidence', '?')})"
            thick = 3
        else:
            # 排除干扰：画绿框 + 虚线效果(模拟)
            color = (0, 255, 0)
            label = "Ignored"
            thick = 2

        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, thick)
        cv2.putText(result_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 3. 保存最终结果
    Path(OUTPUT_PATH).parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(OUTPUT_PATH, result_img)
    print(f"\n✅ 全流程结束！")
    print(f"📂 结果图片已保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()