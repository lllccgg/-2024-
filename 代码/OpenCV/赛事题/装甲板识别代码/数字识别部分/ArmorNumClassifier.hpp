#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "config.hpp"
#include "ArmorBox.hpp"
#include <opencv2/dnn/dnn.hpp>

using namespace std;
using namespace cv;

class ArmorNumClassifier
{
public:
	ArmorNumClassifier();
	~ArmorNumClassifier() = default;
	bool getArmorImg(ArmorBox& armor, const Mat& srcImg); // 获取装甲板图像
	void loadONNXModel(const char* model_path); // 加载ONNX模型
	void getArmorNumByONNX(ArmorBox& armor); // 识别装甲板数字（ONNX）
private:
	Size armorImgSize; // 输入图像尺寸
	Size yoloInputSize; // YOLO输入尺寸
	Mat warpPerspective_src; // 透视变换前的原图
	Mat warpPerspective_dst; // 透视变换后的目标图
	Mat warpPerspective_mat; // 透视变换矩阵
	Point2f srcPoints[4]; // 透视变换原图的目标点（左上 右上 右下 左下）

	dnn::Net onnx_net; // ONNX模型
	bool use_onnx = false; // 是否使用ONNX模型

	vector <string> class_names; // 类别名称映射 {"0", "1", "2", "3", "4", "5", "6", "7", "8"}
	float confidence_threshold; // 置信度阈值

	Mat preprocessForYOLO(const Mat& input_img); // YOLOv8分类预处理
};
