#pragma once
#include <opencv2/opencv.hpp>
#include "LightBar.hpp"
#include "config.hpp"

using namespace cv;
using namespace std;


class ArmorBox
{
public:
	LightBar leftlight; // �������
	LightBar rightlight; // �Ҳ�����
	Point2f center; // װ�װ�����
	float width; // ���
	float height; // �߶�
	float angle; // �Ƕ�
	vector<Point2f> vertices; // �ĸ�����

	Size2f size; // װ�װ�ߴ�
	RotatedRect rotatedrect; // ��ת����
	double distance; // ����
	Vec3d rvec; // ��ת����
	Vec3d tvec; // ƽ������
	double yaw_angle; // ƫ����
	double pitch_angle; // ������
	double roll_angle; // ������

	int armorNum; // װ�װ�����

public:
	ArmorBox(const LightBar& light1, const LightBar& light2);
	~ArmorBox() = default;
	// �ж��Ƿ�Ϊ��Чװ�װ�
	bool isSuitableArmor() const;
	// ��ȡװ�װ��ĸ�����
	void getVertices(Point2f vertices[4]) const;
	// ����װ�װ�����
	void calculateCenter() ;
	// ����װ�װ�ߴ�ͽǶ�
	void calculateGeometry();

	// PnP�������
	bool solvePnP(const Mat& caremaMatrix, const Mat& distCoeffs);
	// ��ȡװ�װ��ĸ��ǵ㣨���ϡ����ϡ����¡����£�
	vector<Point2f> getCornerPoints();
	// ����װ�װ���Ϣ���������̬�ǣ�
	void draw(Mat& img, const Scalar& color = Scalar(0, 0, 255)) const;
	// �����������ƵĽǶȲ�
	float calculateAngleDiff() const;
	// �����������Ƶĳ��ȱ�
	float calculateLengthRatio() const;
	// �����λ�Ƕ�
	float getDeviationAngle() const;
};