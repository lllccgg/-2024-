#pragma once
#include <opencv2/opencv.hpp>
#include "config.hpp"
#include <cmath>

using namespace cv;


class LightBar
{
public:
	RotatedRect rect; // ��ת����
	Point2f center; // ��������
	float length; // ����
	float width; // ���
	float angle; // �Ƕȣ��봹ֱ����ļнǣ�
	float area; // ���

	LightBar() = default;
	LightBar(const RotatedRect& roraatedrect);
	~LightBar() = default;

	// �ж��Ƿ�Ϊ����
	bool isValidLightBar() const;
	// ��ȡ�ĸ�����
	void getVertices(Point2f vertices[4]) const;
};
