#include "BBox.hpp"

// IOU计算
float BBox::iou(const BBox& other) const
{
	// 计算交集矩形边界
	float inter_left = max(this->left(), other.left());
	float inter_top = max(this->top(), other.top());
	float inter_right = min(this->right(), other.right());
	float inter_bottom = min(this->bottom(), other.bottom());
	// 检查是否存在交集
	if (inter_left >= inter_right || inter_top >= inter_bottom)
	{
		return 0.0f;
	}
	// 计算交集面积
	float inter_area = (inter_right - inter_left) * (inter_bottom - inter_top);
	// 计算并集面积
	float union_area = this->area() + other.area() - inter_area;
	// 避免除零错误
	return (union_area > 0) ? (inter_area / union_area) : 0.0f;
}
// GIOU计算（解决了两个目标没有交集时梯度为零的问题）
float BBox::giou(const BBox& other) const
{
	float iou_val = this->iou(other);
	// 计算最小外接矩形
	float convex_left = min(this->left(), other.left());
	float convex_top = min(this->top(), other.top());
	float convex_right = max(this->right(), other.right());
	float convex_bottom = max(this->bottom(), other.bottom());
	// 计算最小外接矩形面积
	float convex_area = (convex_right - convex_left) * (convex_bottom - convex_top);
	// 计算并集面积
	float union_area = this->area() + other.area() - this->intersect(other).area();
	// GIOU = IOU - （最小外接矩形面积 - 并集面积）/ 最小外接矩形面积
	return iou_val - (convex_area - union_area) / convex_area;
}
// DIOU计算（直接优化两个边界框的中心点距离）
float BBox::diou(const BBox& other) const
{
	float iou_val = this->iou(other);
	// 计算中心点距离的平方
	Point2f center1 = this->center();
	Point2f center2 = other.center();
	float center_dist_sq = (center1.x - center2.x) * (center1.x - center2.x) + (center1.y - center2.y) * (center1.y - center2.y);
	// 计算最小外接矩形对角线距离平方
	float convex_left = min(this->left(), other.left());
	float convex_top = min(this->top(), other.top());
	float convex_right = max(this->right(), other.right());
	float convex_bottom = max(this->bottom(), other.bottom());
	float diagonal_sq = (convex_right - convex_left) * (convex_right - convex_left) + (convex_bottom - convex_top) * (convex_bottom - convex_top);
	// DIOU = IOU - 中心距离平方 / 对角线距离平方
	return iou_val - center_dist_sq / diagonal_sq;
}
// CIOU计算（同时考虑重叠面积、中心距离和长宽比）
float BBox::ciou(const BBox& other) const
{
	float diou_val = this->diou(other);
	// 计算长宽比一致性参数v
	float aspect_ratio1 = this->width / this->height;
	float aspect_ratio2 = this->width / this->height;
	float v = (4 / (CV_PI * CV_PI)) * pow(atan(aspect_ratio1) - atan(aspect_ratio2), 2);
	// 计算权重参数α
	float iou_val = this->iou(other);
	float alpha = v / (1 - iou_val + v + 1e-8f); // 避免除零
	// CIOU = DIOU - α * v
	return diou_val - alpha * v;
}

// 计算两个边界框的交集
BBox BBox::intersect(const BBox& other) const
{
	float inter_left = max(this->left(), other.left());
	float inter_top = max(this->top(), other.top());
	float inter_right = min(this->right(), other.right());
	float inter_bottom = min(this->bottom(), other.bottom());
	// 检查是否存在交集
	if (inter_left >= inter_right || inter_top >= inter_bottom)
	{
		return BBox();
	}
	return BBox(inter_left, inter_top, (inter_right - inter_left), (inter_bottom - inter_top));
}
// 计算两个边界框的并集
BBox BBox::unite(const BBox& other) const
{
	float union_left = min(this->left(), other.left());
	float union_top = min(this->top(), other.top());
	float union_right = max(this->right(), other.right());
	float union_bottom = max(this->bottom(), other.bottom());
	return BBox(union_left, union_top, (union_right - union_left), (union_bottom - union_top));
}
// 判断点是否在边界框内
bool BBox::contains(const cv::Point2f& point) const
{
	return point.x >= this->left() && point.x <= this->right() && point.y >= this->top() && point.y <= this->bottom();
}
// 按比例缩放边界框
void BBox::scale(float factor)
{
	Point2f center_point = this->center();
	width *= factor;
	height *= factor;
	x = center_point.x - width / 2.0f;
	y = center_point.y - height / 2.0f;
}
// 向外扩展边界框
void BBox::expand(float margin)
{
	x -= margin;
	y -= margin;
	width += 2 * margin;
	height += 2 * margin;
}