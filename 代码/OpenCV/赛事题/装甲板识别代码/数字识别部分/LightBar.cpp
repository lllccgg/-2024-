#include "LightBar.hpp"


LightBar::LightBar(const RotatedRect& rotatedRect)
{
	rect = rotatedRect;
	center = rotatedRect.center;
	length = max(rotatedRect.size.width, rotatedRect.size.height);
	width = min(rotatedRect.size.width, rotatedRect.size.height);
	
	// 角度标准化处理：确保角度在合理范围内
	angle = rotatedRect.angle;
	// 将角度规范化到[-90, 90]范围
	while (angle > 90) angle -= 180;
	while (angle < -90) angle += 180;

	area = rotatedRect.size.area();
}

void LightBar::getVertices(Point2f vertices[4]) const
{
	rect.points(vertices);  // 获取四个顶点
}

bool LightBar::isValidLightBar() const
{
	float ratio = length / width; // 以长边为长，短边为宽
	float normalizeAngle = abs(angle); // 角度规范化到[0,90]范围  
	// 1.检查长宽比
	if (ratio < Config::MIN_LIGHT_RATIO || ratio > Config::MAX_LIGHT_RATIO) return false;

	// 2.检查面积
	if (area < Config::MIN_LIGHT_AREA || area > Config::MAX_LIGHT_AREA) return false;

	// 3.检查角度
	if (normalizeAngle > 45) normalizeAngle = 90 - normalizeAngle;
	if (normalizeAngle > Config::MAX_ANGLE_DIFF) return false;

	return true;
}
