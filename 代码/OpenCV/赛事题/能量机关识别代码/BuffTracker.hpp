#pragma once
#include <opencv2/opencv.hpp>
#include "config.hpp"
#include "BBox.hpp"
#include <vector>
#include <deque>

using namespace std;
using namespace cv;

// ��Ҷ״̬
enum class FanBladeType
{
	Lighted, // ����������
	UNKNOW // δ֪��δ����
};

// ��Ҷ��ת����
enum class RotationDirection
{
	UNKONW, // δ֪
	CLOCKWISE, // ˳ʱ��
	ANTICLOCKWISE // ��ʱ��
};

// ��Ҷ��
class FanBlade
{
public:
	RotatedRect box; // ��ת����
	FanBladeType state; // ��Ҷ״̬
	float angle; // ��Ҷ�Ƕ�
	Point2f center; // ����
	FanBlade() : state(FanBladeType::UNKNOW), angle(0.0f) {}
	FanBlade(const RotatedRect& rect, FanBladeType type) : box(rect), state(type), angle(rect.angle), center(rect.center) {}
	// ��ȡ��Ҷ�ĳ���ȣ�����Ϊ�����̱�Ϊ��
	float getAspectRatio() const
	{
		float width = box.size.width;
		float height = box.size.height;
		return max(width, height) / min(width, height);
	}
};

// Ŀ��׷����
class BuffTracker
{
public:
	BuffTracker(const BBox& r_box, const BBox& f_box); // ���캯��
	bool update(const Mat& src); // ����ÿһ֡ͼ��
	
	vector<FanBlade> getLightedFanBlades() const
	{
		return lighted_fanblade;
	}
private:
	SimpleTimer timer; // ��ʱ��

	BBox r_box; // ��ǰ֡����R
	Point2f last_center; // ��һ֡����R�����ĵ�
	float radius; // ����R����Ҷ���ĵľ���

	vector<FanBlade> lighted_fanblade; // ��������Ҷ
	FanBlade last_fanblade; // ��һ֡�Ĵ�������Ҷ

	FanBlade target_fanblade; // ��ǰ���ڸ��ٵ�Ŀ����Ҷ
	bool is_tracking; // �Ƿ����ڸ��ٵı�־
	float last_raw_angle = 0.0f; // ��һ֡��ԭʼ�Ƕ�
	float continuous_angle = 0.0f; // �����������Ƕȣ��൱����̼ƣ�����Ԥ�⣬δ��ɣ�
	RotationDirection rotation_direction = RotationDirection::UNKONW; // ��ת����
	deque<float> angle_diff_history; // �洢���N֡�ĽǶȱ仯

	void getMask(const Mat& src, Mat& mask); // ͼ��Ԥ����
	void getAlternateBoxes(const Mat& mask, vector<BBox>& alternateboxes, Mat& img); // Ѱ�Ҳ�ɸѡ����ѡ������R��
	bool compareByCIOU(const vector<BBox>& alternateboxes, BBox& bestbox); // ͨ��CIOU֡��ƥ�䣬�õ����ŵ�����R
	void showR(const BBox& bestbox, Mat& img); // ���ʶ�𵽵�����R

	vector<FanBlade> getFanBlade(const Mat& mask); // ʶ����Ҷ
	void showFan(const vector<FanBlade>& f_boxes, Mat& img); // ���ʶ�𵽵Ĵ�������Ҷ

	void trackFanBlade(); // ����Ŀ����Ҷ
	void updateRotationDirection(); // ������ת����
};