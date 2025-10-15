import cv2
import numpy as np

# --- 1. 参数定义 ---
ONNX_MODEL_PATH = "E:/yolov8-main/runs/detect/SHANZHI_NEW1/weights/best.onnx" # ONNX模型路径
CONFIDENCE_THRESHOLD = 0.45  # 置信度阈值
NMS_THRESHOLD = 0.45       # 非极大值抑制阈值
INPUT_WIDTH = 640          # 模型输入宽度
INPUT_HEIGHT = 640         # 模型输入高度
CLASSES = ['sanji']        # 类别名称

# --- 2. 加载ONNX模型 ---
try:
    net = cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)
    print("ONNX model loaded successfully!")
except cv2.error as e:
    print(f"Error loading ONNX model: {e}")
    exit()

# --- 3. 预处理函数 (信封调整) ---
def preprocess(image):
    h, w, _ = image.shape
    scale = min(INPUT_WIDTH / w, INPUT_HEIGHT / h)
    scaled_w, scaled_h = int(w * scale), int(h * scale)
    scaled_img = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    # 创建一个灰色背景的画布
    top_pad = (INPUT_HEIGHT - scaled_h) // 2
    bottom_pad = INPUT_HEIGHT - scaled_h - top_pad
    left_pad = (INPUT_WIDTH - scaled_w) // 2
    right_pad = INPUT_WIDTH - scaled_w - left_pad
    
    padded_img = cv2.copyMakeBorder(scaled_img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # BGR to RGB, HWC to CHW, and normalize to [0, 1]
    blob = cv2.dnn.blobFromImage(padded_img, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    return blob, scale, left_pad, top_pad

# --- 4. 后处理函数 ---
def postprocess(outputs, scale, left_pad, top_pad):
    # YOLOv8的输出格式是 [batch, 5, num_proposals] -> [cx, cy, w, h, conf]
    # 我们需要将其转置为 [batch, num_proposals, 5]
    outputs = np.transpose(outputs[0])
    
    boxes = []
    confidences = []
    class_ids = []

    for row in outputs:
        confidence = row[4]
        if confidence > CONFIDENCE_THRESHOLD:
            # YOLOv8的输出是 [cx, cy, w, h]
            cx, cy, w, h = row[:4]
            
            # 转换回原始图像坐标
            left = int((cx - w / 2 - left_pad) / scale)
            top = int((cy - h / 2 - top_pad) / scale)
            width = int(w / scale)
            height = int(h / scale)
            
            boxes.append([left, top, width, height])
            confidences.append(float(confidence))
            class_ids.append(0) # 只有一个类别

    # 应用非极大值抑制
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append({
                "box": boxes[i],
                "confidence": confidences[i],
                "class_id": class_ids[i]
            })
    return final_boxes

# --- 5. 主循环：摄像头实时检测 ---
def main():
    cap = cv2.VideoCapture(0) # 打开默认摄像头
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 预处理
        blob, scale, left_pad, top_pad = preprocess(frame)

        # 模型推理
        net.setInput(blob)
        outputs = net.forward()

        # 后处理
        detections = postprocess(outputs, scale, left_pad, top_pad)

        # 绘制结果
        for det in detections:
            box = det['box']
            confidence = det['confidence']
            
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{CLASSES[0]}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示结果
        cv2.imshow("Real-time Sanji Detection", frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()