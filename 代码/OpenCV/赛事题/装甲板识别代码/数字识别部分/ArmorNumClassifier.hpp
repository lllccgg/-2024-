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
	bool getArmorImg(ArmorBox& armor, const Mat& srcImg); // ��ȡװ�װ�ͼ��
	void loadONNXModel(const char* model_path); // ����ONNXģ��
	void getArmorNumByONNX(ArmorBox& armor); // ʶ��װ�װ����֣�ONNX��
private:
	Size armorImgSize; // SVM����ͼ��ߴ�
	Mat warpPerspective_src; // ͸�ӱ任ǰ��ԭͼ
	Mat warpPerspective_dst; // ͸�ӱ任���Ŀ��ͼ
	Mat warpPerspective_mat; // ͸�ӱ任����
	Point2f srcPoints[4]; // ͸�ӱ任ԭͼ��Ŀ��㣨���� ���� ���� ���£�

	dnn::Net onnx_net; // ONNXģ��
	bool use_onnx = false; // �Ƿ�ʹ��ONNXģ��
};