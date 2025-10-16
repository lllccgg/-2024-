#include "BuffTracker.hpp"

BuffTracker::BuffTracker(const BBox& r_box, const BBox& f_box)
{
	BuffTracker::r_box = r_box; // 初始化上一帧中心R
	last_center = r_box.center(); // 初始化上一帧中心R的中心点

	Point2f r_center = r_box.center(); // 中心R的中心点
	Point2f f_center = f_box.center(); // 扇叶的中心点
	radius = norm(r_center - f_center); // 计算中心R到扇叶中心的距离
}
bool BuffTracker::update(const Mat& src) 
{
	Mat img = src.clone();
	Mat mask;

	// 1.图像预处理，得到二值化掩码
	timer.start();
	getMask(src, mask);
	timer.printElapsed("图像预处理");

	// 2.寻找并筛选出候选框（中心R）
	timer.start();
	vector<BBox> alternateboxes;
	getAlternateBoxes(mask, alternateboxes, img);

	// 3.使用CIOU与上一帧中心R进行匹配，得到当前帧中心R
	BBox r_bestbox;
	bool r_success = compareByCIOU(alternateboxes, r_bestbox);
	if (r_success)
	{
		//cout << "success" << endl;
		showR(r_bestbox, img);
	}
	else
	{
		//cout << "false" << endl;
	}
	timer.printElapsed("检测中心R");

	// 4.检测扇叶
	timer.start();
	vector<FanBlade> fan_boxes;
	fan_boxes = getFanBlade(mask);
	if (!fan_boxes.empty())
	{
		//cout << "有检测到扇叶" << endl;
		showFan(fan_boxes, img);
		lighted_fanblade = fan_boxes;
	}
	else
	{
		//cout << "未检测到扇叶" << endl;
	}
	timer.printElapsed("检测扇叶");

	// 5.跟踪目标扇叶并判断旋转方向
	timer.start();
	trackFanBlade(); // 跟踪目标扇叶
	if (is_tracking) // 如果有跟踪目标，开始判断并更新旋转方向
	{
		updateRotationDirection();
	}

	if (rotation_direction == RotationDirection::ANTICLOCKWISE)
	{
		cout << "逆时针旋转" << endl;
	}
	else if (rotation_direction == RotationDirection::CLOCKWISE)
	{
		cout << "顺时针旋转" << endl;
	}
	else
	{
		cout << "未知" << endl;
	}
	timer.printElapsed("跟踪扇叶并判断旋转方向");
	imshow("result", img);
	return r_success;
}

void BuffTracker::getMask(const Mat& src, Mat& mask)
{
	// 1.使用通道相减法进行颜色提取
	Mat binary;
	vector<Mat> channels;
	split(src, channels);

	//这里识别的装甲板颜色定为蓝色
	Mat blueChannel = channels[0] - channels[2];

	// 2.二值化处理
	threshold(blueChannel, binary, Config::BINARY_THRESHHOLD, 255, THRESH_BINARY);

	// 3.形态学操作去噪
	Mat kernel1 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(binary, binary, MORPH_CLOSE, kernel1);

	// 4.膨胀操作
	Mat kernel2 = getStructuringElement(MORPH_DILATE, Size(Config::morph_kernel_size, Config::morph_kernel_size));
	dilate(binary, mask, kernel2, Point(-1, -1), Config::morph_iterations);

	imshow("", mask);
}

void BuffTracker::getAlternateBoxes(const Mat& mask, vector<BBox>& alternateboxes, Mat& img)
{
	vector<vector<Point>> contours;
	// 1.寻找轮廓
	findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	drawContours(img, contours, -1, Scalar(255, 0, 0));
	// 2.遍历所有找到的轮廓
	for (const auto& contour : contours)
	{
		// 3.获取轮廓的最小外接矩形
		Rect rect = boundingRect(contour);
		// 4.面积筛选
		double area = contourArea(contour);
		if (area < Config::r_center_min_area || area > Config::r_center_max_area)
		{
			//cout << "没有通过面积筛选" << endl;
			continue;
		}
		// 5.长宽比筛选
		float aspec_ratio = max(rect.width, rect.height) / min(rect.width, rect.height);
		if (aspec_ratio < Config::r_center_min_ratio || aspec_ratio > Config::r_center_max_ratio)
		{
			//cout << "没有通过长宽比筛选" << endl;
			continue;
		}
		// 6.当前帧中心R中心与上一帧中心R中心的距离筛选（添加上去识别效果不稳定）
		//Point2f current_center = Point2f(rect.x + rect.width / 2.0f, rect.y + rect.height / 2.0f);
		//double distance = norm(current_center - last_center);
		//if (distance > 3 * radius)
		//{
		//	cout << "没有通过距离筛选" << endl;
		//	continue;
		//}
		BBox box(rect, 0); // 将Rect类型转换为BBox类型的
		alternateboxes.push_back(box);
	}
}
bool BuffTracker::compareByCIOU(const vector<BBox>& alternateboxes, BBox& bestbox)
{
	if (alternateboxes.empty())
	{
		//cout << "没有候选框，匹配失败" << endl;
		return false; // 没有候选框，匹配失败
	}
	float best_ciou = -1.0f;
	float bestindex = -1;
	// 1.遍历所有候选框，计算与上一帧中心R的CIOU
	for (size_t i = 0;i < alternateboxes.size();i++)
	{
		float ciou = alternateboxes[i].ciou(r_box);
		if (ciou > best_ciou)
		{
			bestindex = i;
			best_ciou = ciou;
		}
	}
	// 2.如果找到了最优候选框
	if (bestindex != -1)
	{
		bestbox = alternateboxes[bestindex];
		return true;
	}
	//cout << "没有找到最优候选框" << endl;
	return false;
}
void BuffTracker::showR(const BBox& bestbox, Mat& img)
{
	Point2f center = bestbox.center();
	string msg = "(" + to_string((int)center.x) + "," + to_string((int)center.y) + ")";
	circle(img, center, 8, Scalar(0, 255, 0),2);
	putText(img, msg, Point2f(center.x - 10, center.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
}
vector<FanBlade> BuffTracker::getFanBlade(const Mat& mask)
{
	vector<FanBlade> fanblade_result;

	// 1.寻找轮廓
	vector<vector<Point>> contours;
	findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	for (const auto& contour : contours)
	{
		// 2.面积筛选
		double area = contourArea(contour);
		if (area < Config::fan_blade_min_area || area > Config::fan_blade_max_area)
		{
			//cout << "扇叶没通过面积筛选" << endl;
			continue;
		}
		// 3.获取旋转矩阵筛选
		RotatedRect fan_rrect = minAreaRect(contour);
		// 4.长宽比筛选（采用旋转矩阵，因为扇叶是旋转物体，用旋转矩阵更合适）
		float w = fan_rrect.size.width;
		float h = fan_rrect.size.height;
		float aspect_ratio = max(w, h) / min(w, h);
		if (aspect_ratio < Config::fan_blade_min_ratio || aspect_ratio > Config::fan_blade_max_ratio)
		{
			//cout << "扇叶未通过长宽比筛选" << endl;
			continue;
		}
		fanblade_result.push_back(FanBlade(fan_rrect, FanBladeType::Lighted));
	}
	return fanblade_result;
}
void BuffTracker::showFan(const vector<FanBlade>& f_boxes, Mat& img)
{
	for (const auto& f_box : f_boxes)
	{
		// 获取扇叶的旋转矩阵
		RotatedRect f_rrect = f_box.box; 
		// 获取旋转矩形的四个顶点
		Point2f vertices[4];
		f_rrect.points(vertices);
		// 循环绘制四条边
		for (int i = 0; i < 4; i++)
		{
			line(img, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 255), 2);
		}
	}
}
void BuffTracker::trackFanBlade()
{
	if (lighted_fanblade.empty()) // 如果没有检测到待击打扇叶
	{
		is_tracking = false;
		return;
	}
	if (!is_tracking) // 如果之前没有目标扇叶，就挑选检测的待击打扇叶中的一个
	{
		target_fanblade = lighted_fanblade[0];
		is_tracking = true;
	}
	else // 如果已经有目标扇叶，寻找最近的扇叶作为当前帧的追踪扇叶
	{
		float min_dist = -1.0f;
		int best_indest = -1;
		for (size_t i = 0; i < lighted_fanblade.size(); i++)
		{
			// 计算与上一帧目标扇叶的距离
			Point2f diff = lighted_fanblade[i].center - target_fanblade.center;
			float distance = norm(diff);
			if (best_indest == -1 || distance < min_dist)
			{
				min_dist = distance;
				best_indest = i;
			}
		}
		// 更新目标
		target_fanblade = lighted_fanblade[best_indest];
	}
}
void BuffTracker::updateRotationDirection()
{
	// 1.计算当前的原始角度（相对于中心R）
	float current_raw_angle = atan2((target_fanblade.center.y - r_box.center().y), target_fanblade.center.x - r_box.center().x); 
	// 如果是第一次，初始化角度
	if (!angle_diff_history.empty())
	{
		last_raw_angle = current_raw_angle;
		continuous_angle = current_raw_angle;
		angle_diff_history.push_back(0.0f);
	}
	// 2.角度解算（避免-179°到1°这种情况的角度突变）
	float angle_diff = current_raw_angle - last_raw_angle; // 计算两帧之间的角度变化
	if (angle_diff > CV_PI)
	{
		angle_diff -= 2 * CV_PI;
	}
	else if (angle_diff < -CV_PI)
	{
		angle_diff += 2 * CV_PI;
	}
	// 更新连续角度和上一帧角度
	continuous_angle += angle_diff;
	last_raw_angle = current_raw_angle;
	// 3.将本次的角度变化量存入历史队列
	angle_diff_history.push_back(angle_diff);
	if (angle_diff_history.size() > 10)
	{
		angle_diff_history.pop_front(); // 保持队列大小
	}
	// 计算历史队列中所有角度变化的总和
	float total_diff = 0.0f;
	for (const auto& diff : angle_diff_history)
	{
		total_diff += diff;
	}
	// 根据总和的符号判断方向,设置一个小的阈值避免噪声干扰
	if (total_diff > 0.01)
	{
		rotation_direction = RotationDirection::ANTICLOCKWISE; // 记为逆时针方向
	}
	else if (total_diff < -0.01)
	{
		rotation_direction = RotationDirection::CLOCKWISE; // 记为顺时针方向
	}
	else
	{
		rotation_direction = RotationDirection::UNKONW; // 记为未知
	}
}