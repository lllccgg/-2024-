#include "LightBar.hpp"


LightBar::LightBar(const RotatedRect& rotatedRect)
{
	rect = rotatedRect;
	center = rotatedRect.center;
	length = max(rotatedRect.size.width, rotatedRect.size.height);
	width = min(rotatedRect.size.width, rotatedRect.size.height);
	
	// �Ƕȱ�׼������ȷ���Ƕ��ں���Χ��
	angle = rotatedRect.angle;
	// ���Ƕȹ淶����[-90, 90]��Χ
	while (angle > 90) angle -= 180;
	while (angle < -90) angle += 180;

	area = rotatedRect.size.area();
}

void LightBar::getVertices(Point2f vertices[4]) const
{
	rect.points(vertices);  // ��ȡ�ĸ�����
}

bool LightBar::isValidLightBar() const
{
	float ratio = length / width; // �Գ���Ϊ�����̱�Ϊ��
	float normalizeAngle = abs(angle); // �Ƕȹ淶����[0,90]��Χ  
	// 1.��鳤���
	if (ratio < Config::MIN_LIGHT_RATIO || ratio > Config::MAX_LIGHT_RATIO) return false;

	// 2.������
	if (area < Config::MIN_LIGHT_AREA || area > Config::MAX_LIGHT_AREA) return false;

	// 3.���Ƕ�
	if (normalizeAngle > 45) normalizeAngle = 90 - normalizeAngle;
	if (normalizeAngle > Config::MAX_ANGLE_DIFF) return false;

	return true;
}
