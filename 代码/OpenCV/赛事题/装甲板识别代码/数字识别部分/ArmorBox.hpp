#pragma once
#include <opencv2/opencv.hpp>
#include "LightBar.hpp"
#include "config.hpp"

using namespace cv;
using namespace std;


class ArmorBox
{
public:
	LightBar leftlight; // 左侧条灯
	LightBar rightlight; // 右侧条灯
	Point2f center; // 装甲板中心
	float width; // 宽度
	float height; // 高度
	float angle; // 角度
	vector<Point2f> vertices; // 四个顶点

	Size2f size; // 装甲板尺寸
	RotatedRect rotatedrect; // 旋转矩阵
	double distance; // 距离
	Vec3d rvec; // 旋转向量
	Vec3d tvec; // 平移向量
	double yaw_angle; // 偏航角
	double pitch_angle; // 俯仰角
	double roll_angle; // 翻滚角

	int armorNum; // 装甲板数字

public:
	ArmorBox(const LightBar& light1, const LightBar& light2);
	~ArmorBox() = default;
	// 判断是否为有效装甲板
	bool isSuitableArmor() const;
	// 获取装甲板四个顶点
	void getVertices(Point2f vertices[4]) const;
	// 计算装甲板中心
	void calculateCenter() ;
	// 计算装甲板尺寸和角度
	void calculateGeometry();

	// PnP距离解算
	bool solvePnP(const Mat& caremaMatrix, const Mat& distCoeffs);
	// 获取装甲板四个角点（左上、右上、右下、左下）
	vector<Point2f> getCornerPoints();
	// 绘制装甲板信息（距离和姿态角）
	void draw(Mat& img, const Scalar& color = Scalar(0, 0, 255)) const;
	// 计算两个条灯的角度差
	float calculateAngleDiff() const;
	// 计算两个条灯的长度比
	float calculateLengthRatio() const;
	// 计算错位角度
	float getDeviationAngle() const;
};