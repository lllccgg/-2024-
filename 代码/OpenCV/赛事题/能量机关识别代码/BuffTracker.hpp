#pragma once
#include <opencv2/opencv.hpp>
#include "config.hpp"
#include "BBox.hpp"
#include <vector>
#include <deque>

using namespace std;
using namespace cv;

// 扇叶状态
enum class FanBladeType
{
	Lighted, // 待击打（亮起）
	UNKNOW // 未知（未亮起）
};

// 扇叶旋转方向
enum class RotationDirection
{
	UNKONW, // 未知
	CLOCKWISE, // 顺时针
	ANTICLOCKWISE // 逆时针
};

// 扇叶类
class FanBlade
{
public:
	RotatedRect box; // 旋转矩阵
	FanBladeType state; // 扇叶状态
	float angle; // 扇叶角度
	Point2f center; // 中心
	FanBlade() : state(FanBladeType::UNKNOW), angle(0.0f) {}
	FanBlade(const RotatedRect& rect, FanBladeType type) : box(rect), state(type), angle(rect.angle), center(rect.center) {}
	// 获取扇叶的长宽比（长边为长，短边为宽）
	float getAspectRatio() const
	{
		float width = box.size.width;
		float height = box.size.height;
		return max(width, height) / min(width, height);
	}
};

// 目标追踪类
class BuffTracker
{
public:
	BuffTracker(const BBox& r_box, const BBox& f_box); // 构造函数
	bool update(const Mat& src); // 处理每一帧图像
	
	vector<FanBlade> getLightedFanBlades() const
	{
		return lighted_fanblade;
	}
private:
	SimpleTimer timer; // 计时器

	BBox r_box; // 当前帧中心R
	Point2f last_center; // 上一帧中心R的中心点
	float radius; // 中心R到扇叶中心的距离

	vector<FanBlade> lighted_fanblade; // 待击打扇叶
	FanBlade last_fanblade; // 上一帧的待击打扇叶

	FanBlade target_fanblade; // 当前正在跟踪的目标扇叶
	bool is_tracking; // 是否正在跟踪的标志
	float last_raw_angle = 0.0f; // 上一帧的原始角度
	float continuous_angle = 0.0f; // 解算后的连续角度，相当于里程计（用于预测，未完成）
	RotationDirection rotation_direction = RotationDirection::UNKONW; // 旋转方向
	deque<float> angle_diff_history; // 存储最近N帧的角度变化

	void getMask(const Mat& src, Mat& mask); // 图像预处理
	void getAlternateBoxes(const Mat& mask, vector<BBox>& alternateboxes, Mat& img); // 寻找并筛选出候选框（中心R）
	bool compareByCIOU(const vector<BBox>& alternateboxes, BBox& bestbox); // 通过CIOU帧间匹配，得到最优的中心R
	void showR(const BBox& bestbox, Mat& img); // 标出识别到的中心R

	vector<FanBlade> getFanBlade(const Mat& mask); // 识别扇叶
	void showFan(const vector<FanBlade>& f_boxes, Mat& img); // 标出识别到的待击打扇叶

	void trackFanBlade(); // 跟踪目标扇叶
	void updateRotationDirection(); // 更新旋转方向
};