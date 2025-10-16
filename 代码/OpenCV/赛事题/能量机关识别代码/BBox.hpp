#pragma once
#pragma once
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace cv;

class BBox
{
public:
	float x, y, width, height; // 边界框参数：左上角坐标、宽和高
	//float confidence; // 置信度分数
	int class_id; // 0-中心R，1-扇叶

	// 构造函数
	BBox() : x(0),y(0),width(0),height(0),class_id(-1) {}
	BBox(float x,float y,float width,float height,int id = -1) 
		: x(x),y(y),width(width),height(height),class_id(id){}
	BBox(const Rect&rect,int id = -1)
		: x(static_cast<float>(rect.x)),y(static_cast<float>(rect.y)),
		width(static_cast<float>(rect.width)), height(static_cast<float>(rect.height)), class_id(id){ }

	// 基础几何计算
	inline float area() const
	{
		return width * height;
	}
	inline Point2f center() const
	{
		return Point2f(x + width / 2.0f, y + height / 2.0f);
	}
	inline float left() const { return x; }
	inline float right() const { return x + width; }
	inline float top() const { return y; }
	inline float bottom() const { return y + height; }

	Rect toRect() const
	{
		return Rect(static_cast<float>(x), static_cast<float>(y), static_cast<float>(width), static_cast<float>(height));
	}
	
	// IOU计算
	float iou(const BBox& ohter) const;
	float giou(const BBox& other) const;
	float diou(const BBox& other) const;
	float ciou(const BBox& other) const;

	// 一些函数
	bool contains(const Point2f& point) const; // 判断点是否在边界框内
	BBox intersect(const BBox& ohter) const; // 计算两个边界框的交集
	BBox unite(const BBox& other) const; // 计算两个边界框的并集
	void scale(float factor); // 按比例缩放边界框
	void expand(float margin); // 向外扩展边界框
};