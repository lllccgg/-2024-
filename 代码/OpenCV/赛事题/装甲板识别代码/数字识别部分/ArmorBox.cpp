#include "ArmorBox.hpp"
#include "config.hpp"
#include <cmath>

ArmorBox::ArmorBox(const LightBar& light1, const LightBar& light2)
{
	// 根据x坐标确定左右侧的条灯
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
	// 计算装甲板几何参数
	calculateCenter();
	calculateGeometry();
	rotatedrect = RotatedRect(center,Size2f(Config::ARMOR_WIDTH,Config::LIGHT_HEIGHT), angle);
}

void ArmorBox::calculateCenter()
{
	// 装甲板的中心为两个条灯中心的中点
	center.x = (leftlight.center.x + rightlight.center.x) / 2.0f;
	center.y = (leftlight.center.y + rightlight.center.y) / 2.0f;
}

void ArmorBox::calculateGeometry()
{
	// 计算装甲板宽度（两个条灯中心点之间的距离）
	width = sqrt(pow(leftlight.center.x - rightlight.center.x, 2) + pow(leftlight.center.y - rightlight.center.y, 2));
	// 装甲板高度取两个条灯长度的平均值
	height = (leftlight.length + rightlight.length) / 2.0f;
	// 装甲板角度为两个条灯角度的平均值
	angle = (leftlight.angle + rightlight.angle) / 2.0f;
	// 计算装甲板的四个顶点
	vertices.resize(4);
	// 基于中心点和尺寸来计算顶点（以中心点为原点）
	float halfwidth = width / 2.0f;
	float halfheight = height / 2.0f;
	// 考虑角度旋转
	float cosAngle = cos(angle / 180.0f * CV_PI);
	float sinAngle = sin(angle / 180.0f * CV_PI);
	vertices[0] = Point2f(center.x - halfwidth * cosAngle + halfheight * sinAngle, center.y - halfheight * cosAngle - halfwidth * sinAngle); // 左上顶点
	vertices[1] = Point2f(center.x + halfwidth * cosAngle + halfheight * sinAngle, center.y - halfheight * cosAngle + halfwidth * sinAngle); // 右上顶点
	vertices[2] = Point2f(center.x + halfwidth * cosAngle - halfheight * sinAngle, center.y + halfwidth * sinAngle + halfheight * cosAngle); // 右下顶点
	vertices[3] = Point2f(center.x - halfwidth * cosAngle - halfheight * sinAngle, center.y - halfwidth * sinAngle + halfheight * cosAngle); // 左下顶点
}

bool ArmorBox::isSuitableArmor() const
{
	// 1.角度差检测
	float angleDiff = calculateAngleDiff();
	if (angleDiff > Config::MAX_ARMOR_ANGLE_DIFF) return false;
	// 2.长度比检查（两个条灯的长度比）
	float lengthRatio = calculateLengthRatio();
	if (lengthRatio > Config::MAX_LENGTH_RATIO) return false;
	// 3.错位角检查
	float displacementAngle = getDeviationAngle();
	if (displacementAngle > Config::MAX_DISPLACEMENT_ANGLE) return false;
	// 4.宽高比检查（装甲板是横向的）
	float aspectRatio = width / height;
	if (aspectRatio < Config::MIN_ARMOR_ASPECT_RATIO || aspectRatio > Config::MAX_ARMOR_ASPECT_RATIO) return false;

	return true;
}

float ArmorBox::calculateAngleDiff() const
{
	float diff = abs(leftlight.angle - rightlight.angle);
	// 处理角度跨越0度的情况
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
	// 计算两个条灯中心连线与水平线的夹角
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
	rotatedrect.points(corners); // 获取装甲板四个角点
	vector<Point2f> orderedcorners; // 用于排序后的角点
	Point2f topLeft, topRight, buttomRight, buttomLeft; // 四个排列后的角点
	for (const auto& corner : corners)
	{
		if (corner.x <= center.x && corner.y <= center.y)
		{
			topLeft = corner; // 左上
			//cout << "左上角点" << corner << endl;

		}
		else if (corner.x > center.x && corner.y < center.y)
		{
			topRight = corner; // 右上
			//cout << "右上角点" << corner << endl;
		}
		else if (corner.x > center.x && corner.y > center.y)
		{
			buttomRight = corner; // 右下
			//cout << "右下角点" << corner << endl;
		}
		else if (corner.x < center.x && corner.y > center.y)
		{
			buttomLeft = corner; // 左下
			//cout << "左下角点" << corner << endl;
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

	//cout << "左上角点" << topLeft << endl;
	//cout << "右上角点" << topRight << endl;
	//cout << "右下角点" << buttomRight << endl;
	//cout << "左下角点" << buttomLeft << endl;

	return orderedcorners;
}

bool ArmorBox::solvePnP(const Mat& caremaMatrix, const Mat& distCoeffs)
{
	// 装甲板三维图像坐标（以装甲板中心为原点）
	vector<Point3f> objPoints;
	objPoints.push_back(Point3f(-Config::ARMOR_WIDTH / 2.0, -Config::LIGHT_HEIGHT / 2.0, 0.0)); // 左上
	objPoints.push_back(Point3f(Config::ARMOR_WIDTH / 2.0, -Config::LIGHT_HEIGHT / 2.0, 0.0)); // 右上
	objPoints.push_back(Point3f(Config::ARMOR_WIDTH / 2.0, Config::LIGHT_HEIGHT / 2.0, 0.0)); // 右下
	objPoints.push_back(Point3f(-Config::ARMOR_WIDTH / 2.0, Config::LIGHT_HEIGHT / 2.0, 0.0)); // 左下
	// 装甲板二维图像坐标
	vector<Point2f> imgPoints = getCornerPoints();


	// 使用PnP算法进行解算
	bool success = cv::solvePnP(objPoints, imgPoints, caremaMatrix, distCoeffs, rvec, tvec);
	if (success)
	{
		//cout << "tvec[0]:" << tvec[0] << endl;
		//cout << "tvec[1]:" << tvec[1] << endl;
		//cout << "tvec[2]:" << tvec[2] << endl;
		// 距离就是平移向量的模长
		//distance = norm(tvec);
		distance = tvec[2];
		// 偏航角、俯仰角和翻滚角（rvec变为旋转矩阵rmat，再用反三角函数解算）
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
	// 显示距离和角度
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