#include "BBox.hpp"

// IOU����
float BBox::iou(const BBox& other) const
{
	// ���㽻�����α߽�
	float inter_left = max(this->left(), other.left());
	float inter_top = max(this->top(), other.top());
	float inter_right = min(this->right(), other.right());
	float inter_bottom = min(this->bottom(), other.bottom());
	// ����Ƿ���ڽ���
	if (inter_left >= inter_right || inter_top >= inter_bottom)
	{
		return 0.0f;
	}
	// ���㽻�����
	float inter_area = (inter_right - inter_left) * (inter_bottom - inter_top);
	// ���㲢�����
	float union_area = this->area() + other.area() - inter_area;
	// ����������
	return (union_area > 0) ? (inter_area / union_area) : 0.0f;
}
// GIOU���㣨���������Ŀ��û�н���ʱ�ݶ�Ϊ������⣩
float BBox::giou(const BBox& other) const
{
	float iou_val = this->iou(other);
	// ������С��Ӿ���
	float convex_left = min(this->left(), other.left());
	float convex_top = min(this->top(), other.top());
	float convex_right = max(this->right(), other.right());
	float convex_bottom = max(this->bottom(), other.bottom());
	// ������С��Ӿ������
	float convex_area = (convex_right - convex_left) * (convex_bottom - convex_top);
	// ���㲢�����
	float union_area = this->area() + other.area() - this->intersect(other).area();
	// GIOU = IOU - ����С��Ӿ������ - ���������/ ��С��Ӿ������
	return iou_val - (convex_area - union_area) / convex_area;
}
// DIOU���㣨ֱ���Ż������߽������ĵ���룩
float BBox::diou(const BBox& other) const
{
	float iou_val = this->iou(other);
	// �������ĵ�����ƽ��
	Point2f center1 = this->center();
	Point2f center2 = other.center();
	float center_dist_sq = (center1.x - center2.x) * (center1.x - center2.x) + (center1.y - center2.y) * (center1.y - center2.y);
	// ������С��Ӿ��ζԽ��߾���ƽ��
	float convex_left = min(this->left(), other.left());
	float convex_top = min(this->top(), other.top());
	float convex_right = max(this->right(), other.right());
	float convex_bottom = max(this->bottom(), other.bottom());
	float diagonal_sq = (convex_right - convex_left) * (convex_right - convex_left) + (convex_bottom - convex_top) * (convex_bottom - convex_top);
	// DIOU = IOU - ���ľ���ƽ�� / �Խ��߾���ƽ��
	return iou_val - center_dist_sq / diagonal_sq;
}
// CIOU���㣨ͬʱ�����ص���������ľ���ͳ���ȣ�
float BBox::ciou(const BBox& other) const
{
	float diou_val = this->diou(other);
	// ���㳤���һ���Բ���v
	float aspect_ratio1 = this->width / this->height;
	float aspect_ratio2 = this->width / this->height;
	float v = (4 / (CV_PI * CV_PI)) * pow(atan(aspect_ratio1) - atan(aspect_ratio2), 2);
	// ����Ȩ�ز�����
	float iou_val = this->iou(other);
	float alpha = v / (1 - iou_val + v + 1e-8f); // �������
	// CIOU = DIOU - �� * v
	return diou_val - alpha * v;
}

// ���������߽��Ľ���
BBox BBox::intersect(const BBox& other) const
{
	float inter_left = max(this->left(), other.left());
	float inter_top = max(this->top(), other.top());
	float inter_right = min(this->right(), other.right());
	float inter_bottom = min(this->bottom(), other.bottom());
	// ����Ƿ���ڽ���
	if (inter_left >= inter_right || inter_top >= inter_bottom)
	{
		return BBox();
	}
	return BBox(inter_left, inter_top, (inter_right - inter_left), (inter_bottom - inter_top));
}
// ���������߽��Ĳ���
BBox BBox::unite(const BBox& other) const
{
	float union_left = min(this->left(), other.left());
	float union_top = min(this->top(), other.top());
	float union_right = max(this->right(), other.right());
	float union_bottom = max(this->bottom(), other.bottom());
	return BBox(union_left, union_top, (union_right - union_left), (union_bottom - union_top));
}
// �жϵ��Ƿ��ڱ߽����
bool BBox::contains(const cv::Point2f& point) const
{
	return point.x >= this->left() && point.x <= this->right() && point.y >= this->top() && point.y <= this->bottom();
}
// ���������ű߽��
void BBox::scale(float factor)
{
	Point2f center_point = this->center();
	width *= factor;
	height *= factor;
	x = center_point.x - width / 2.0f;
	y = center_point.y - height / 2.0f;
}
// ������չ�߽��
void BBox::expand(float margin)
{
	x -= margin;
	y -= margin;
	width += 2 * margin;
	height += 2 * margin;
}