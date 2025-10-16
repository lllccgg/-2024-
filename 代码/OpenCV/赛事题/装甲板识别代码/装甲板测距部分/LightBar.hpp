#pragma once
#include <opencv2/opencv.hpp>
#include "config.hpp"
#include <cmath>

using namespace cv;


class LightBar
{
public:
	RotatedRect rect; // 旋转矩阵
	Point2f center; // 条灯中心
	float length; // 长度
	float width; // 宽度
	float angle; // 角度（与垂直方向的夹角）
	float area; // 面积

	LightBar() = default;
	LightBar(const RotatedRect& roraatedrect);
	~LightBar() = default;

	// 判断是否为条灯
	bool isValidLightBar() const;
	// 获取四个顶点
	void getVertices(Point2f vertices[4]) const;
};
