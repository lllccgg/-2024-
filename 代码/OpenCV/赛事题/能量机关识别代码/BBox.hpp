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
	float x, y, width, height; // �߽����������Ͻ����ꡢ��͸�
	//float confidence; // ���Ŷȷ���
	int class_id; // 0-����R��1-��Ҷ

	// ���캯��
	BBox() : x(0),y(0),width(0),height(0),class_id(-1) {}
	BBox(float x,float y,float width,float height,int id = -1) 
		: x(x),y(y),width(width),height(height),class_id(id){}
	BBox(const Rect&rect,int id = -1)
		: x(static_cast<float>(rect.x)),y(static_cast<float>(rect.y)),
		width(static_cast<float>(rect.width)), height(static_cast<float>(rect.height)), class_id(id){ }

	// �������μ���
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
	
	// IOU����
	float iou(const BBox& ohter) const;
	float giou(const BBox& other) const;
	float diou(const BBox& other) const;
	float ciou(const BBox& other) const;

	// һЩ����
	bool contains(const Point2f& point) const; // �жϵ��Ƿ��ڱ߽����
	BBox intersect(const BBox& ohter) const; // ���������߽��Ľ���
	BBox unite(const BBox& other) const; // ���������߽��Ĳ���
	void scale(float factor); // ���������ű߽��
	void expand(float margin); // ������չ�߽��
};