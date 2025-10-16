#include "ArmorBox.hpp"
#include "config.hpp"
#include <cmath>

ArmorBox::ArmorBox(const LightBar& light1, const LightBar& light2)
{
	// ����x����ȷ�����Ҳ������
	if (light1.center.x < light2.center.x)
	{
		leftlight = light1;
		rightlight = light2;
	}
	else
	{
		leftlight = light2;
		rightlight = light1;
	}
	// ����װ�װ弸�β���
	calculateCenter();
	calculateGeometry();
	rotatedrect = RotatedRect(center,Size2f(Config::ARMOR_WIDTH,Config::LIGHT_HEIGHT), angle);
}

void ArmorBox::calculateCenter()
{
	// װ�װ������Ϊ�����������ĵ��е�
	center.x = (leftlight.center.x + rightlight.center.x) / 2.0f;
	center.y = (leftlight.center.y + rightlight.center.y) / 2.0f;
}

void ArmorBox::calculateGeometry()
{
	// ����װ�װ��ȣ������������ĵ�֮��ľ��룩
	width = sqrt(pow(leftlight.center.x - rightlight.center.x, 2) + pow(leftlight.center.y - rightlight.center.y, 2));
	// װ�װ�߶�ȡ�������Ƴ��ȵ�ƽ��ֵ
	height = (leftlight.length + rightlight.length) / 2.0f;
	// װ�װ�Ƕ�Ϊ�������ƽǶȵ�ƽ��ֵ
	angle = (leftlight.angle + rightlight.angle) / 2.0f;
	// ����װ�װ���ĸ�����
	vertices.resize(4);
	// �������ĵ�ͳߴ������㶥�㣨�����ĵ�Ϊԭ�㣩
	float halfwidth = width / 2.0f;
	float halfheight = height / 2.0f;
	// ���ǽǶ���ת
	float cosAngle = cos(angle / 180.0f * CV_PI);
	float sinAngle = sin(angle / 180.0f * CV_PI);
	vertices[0] = Point2f(center.x - halfwidth * cosAngle + halfheight * sinAngle, center.y - halfheight * cosAngle - halfwidth * sinAngle); // ���϶���
	vertices[1] = Point2f(center.x + halfwidth * cosAngle + halfheight * sinAngle, center.y - halfheight * cosAngle + halfwidth * sinAngle); // ���϶���
	vertices[2] = Point2f(center.x + halfwidth * cosAngle - halfheight * sinAngle, center.y + halfwidth * sinAngle + halfheight * cosAngle); // ���¶���
	vertices[3] = Point2f(center.x - halfwidth * cosAngle - halfheight * sinAngle, center.y - halfwidth * sinAngle + halfheight * cosAngle); // ���¶���
}

bool ArmorBox::isSuitableArmor() const
{
	// 1.�ǶȲ���
	float angleDiff = calculateAngleDiff();
	if (angleDiff > Config::MAX_ARMOR_ANGLE_DIFF) return false;
	// 2.���ȱȼ�飨�������Ƶĳ��ȱȣ�
	float lengthRatio = calculateLengthRatio();
	if (lengthRatio > Config::MAX_LENGTH_RATIO) return false;
	// 3.��λ�Ǽ��
	float displacementAngle = getDeviationAngle();
	if (displacementAngle > Config::MAX_DISPLACEMENT_ANGLE) return false;
	// 4.��߱ȼ�飨װ�װ��Ǻ���ģ�
	float aspectRatio = width / height;
	if (aspectRatio < Config::MIN_ARMOR_ASPECT_RATIO || aspectRatio > Config::MAX_ARMOR_ASPECT_RATIO) return false;

	return true;
}

float ArmorBox::calculateAngleDiff() const
{
	float diff = abs(leftlight.angle - rightlight.angle);
	// ����Ƕȿ�Խ0�ȵ����
	if (diff > 90) diff = 180 - diff;
	return diff;
}

float ArmorBox::calculateLengthRatio() const
{
	float maxlength = max(leftlight.length, rightlight.length);
	float minlength = min(leftlight.length, rightlight.length);
	return (maxlength - minlength) / maxlength;
}

float ArmorBox::getDeviationAngle() const
{
	// ����������������������ˮƽ�ߵļн�
	double dx = leftlight.center.x - rightlight.center.x;
	double dy = leftlight.center.y - rightlight.center.y;
	float angle = atan2(abs(dy), abs(dx)) * 180.0f / CV_PI;
	return angle;
}

void ArmorBox::getVertices(Point2f vertices[4]) const
{
	for (int i = 0; i < 4; i++)
	{
		vertices[i] = this->vertices[i];
	}
}

vector<Point2f> ArmorBox::getCornerPoints()
{
	vector<Point2f> corners;
	rotatedrect.points(corners); // ��ȡװ�װ��ĸ��ǵ�
	vector<Point2f> orderedcorners; // ���������Ľǵ�
	Point2f topLeft, topRight, buttomRight, buttomLeft; // �ĸ����к�Ľǵ�
	for (const auto& corner : corners)
	{
		if (corner.x <= center.x && corner.y <= center.y)
		{
			topLeft = corner; // ����
			//cout << "���Ͻǵ�" << corner << endl;

		}
		else if (corner.x > center.x && corner.y < center.y)
		{
			topRight = corner; // ����
			//cout << "���Ͻǵ�" << corner << endl;
		}
		else if (corner.x > center.x && corner.y > center.y)
		{
			buttomRight = corner; // ����
			//cout << "���½ǵ�" << corner << endl;
		}
		else if (corner.x < center.x && corner.y > center.y)
		{
			buttomLeft = corner; // ����
			//cout << "���½ǵ�" << corner << endl;
		}
	}
	orderedcorners.push_back(topLeft);
	orderedcorners.push_back(topRight);
	orderedcorners.push_back(buttomRight);
	orderedcorners.push_back(buttomLeft);

	//cout << orderedcorners[0] << endl;
	//cout << orderedcorners[1] << endl;
	//cout << orderedcorners[2] << endl;
	//cout << orderedcorners[3] << endl;

	//cout << "���Ͻǵ�" << topLeft << endl;
	//cout << "���Ͻǵ�" << topRight << endl;
	//cout << "���½ǵ�" << buttomRight << endl;
	//cout << "���½ǵ�" << buttomLeft << endl;

	return orderedcorners;
}

bool ArmorBox::solvePnP(const Mat& caremaMatrix, const Mat& distCoeffs)
{
	// װ�װ���άͼ�����꣨��װ�װ�����Ϊԭ�㣩
	vector<Point3f> objPoints;
	objPoints.push_back(Point3f(-Config::ARMOR_WIDTH / 2.0, -Config::LIGHT_HEIGHT / 2.0, 0.0)); // ����
	objPoints.push_back(Point3f(Config::ARMOR_WIDTH / 2.0, -Config::LIGHT_HEIGHT / 2.0, 0.0)); // ����
	objPoints.push_back(Point3f(Config::ARMOR_WIDTH / 2.0, Config::LIGHT_HEIGHT / 2.0, 0.0)); // ����
	objPoints.push_back(Point3f(-Config::ARMOR_WIDTH / 2.0, Config::LIGHT_HEIGHT / 2.0, 0.0)); // ����
	// װ�װ��άͼ������
	vector<Point2f> imgPoints = getCornerPoints();


	// ʹ��PnP�㷨���н���
	bool success = cv::solvePnP(objPoints, imgPoints, caremaMatrix, distCoeffs, rvec, tvec);
	if (success)
	{
		//cout << "tvec[0]:" << tvec[0] << endl;
		//cout << "tvec[1]:" << tvec[1] << endl;
		//cout << "tvec[2]:" << tvec[2] << endl;
		// �������ƽ��������ģ��
		//distance = norm(tvec);
		distance = tvec[2];
		// ƫ���ǡ������Ǻͷ����ǣ�rvec��Ϊ��ת����rmat�����÷����Ǻ������㣩
		//Mat rmat;
		//Rodrigues(rvec, rmat);
		//yaw_angle = atan2(rmat.at<double>(0, 2), rmat.at<double>(2, 2)) * 57.3;
		//pitch_angle = -asin(rmat.at<double>(1, 2)) * 57.3;
		//roll_angle = atan2(rmat.at<double>(1, 0), rmat.at<double>(1, 1)) * 57.3;

		//cout << "rvec: " << rvec.t() << endl;
		//cout << "tvec: " << tvec.t() << endl;
		//double d_norm = norm(tvec);
		//double z = tvec[2];
		//cout << "norm(tvec) = " << d_norm << " (units same as objPoints)" << endl;
		//cout << "tvec.z = " << z << endl;

		//waitKey(0);
	}
	return success;
}

void ArmorBox::draw(Mat& img, const Scalar& color) const
{
	// ��ʾ����ͽǶ�
	if (distance > 0)
	{
		string distText = "distance:" + to_string(distance);
		//string yawText = "yaw:" + to_string(yaw_angle);
		//string pitchText = "pitch:" + to_string(pitch_angle);
		//string rollText = "roll:" + to_string(roll_angle);
		putText(img, distText, center + Point2f(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255));
		//putText(img, yawText, center + Point2f(10, 40), FONT_HERSHEY_SIMPLEX, 0.5, color);
		//putText(img, pitchText, center + Point2f(10, 60), FONT_HERSHEY_SIMPLEX, 0.5, color);
		//putText(img, rollText, center + Point2f(10, 80), FONT_HERSHEY_SIMPLEX, 0.5, color);
	}
}