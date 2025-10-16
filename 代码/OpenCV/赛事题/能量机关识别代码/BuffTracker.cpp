#include "BuffTracker.hpp"

BuffTracker::BuffTracker(const BBox& r_box, const BBox& f_box)
{
	BuffTracker::r_box = r_box; // ��ʼ����һ֡����R
	last_center = r_box.center(); // ��ʼ����һ֡����R�����ĵ�

	Point2f r_center = r_box.center(); // ����R�����ĵ�
	Point2f f_center = f_box.center(); // ��Ҷ�����ĵ�
	radius = norm(r_center - f_center); // ��������R����Ҷ���ĵľ���
}
bool BuffTracker::update(const Mat& src) 
{
	Mat img = src.clone();
	Mat mask;

	// 1.ͼ��Ԥ�����õ���ֵ������
	timer.start();
	getMask(src, mask);
	timer.printElapsed("ͼ��Ԥ����");

	// 2.Ѱ�Ҳ�ɸѡ����ѡ������R��
	timer.start();
	vector<BBox> alternateboxes;
	getAlternateBoxes(mask, alternateboxes, img);

	// 3.ʹ��CIOU����һ֡����R����ƥ�䣬�õ���ǰ֡����R
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
	timer.printElapsed("�������R");

	// 4.�����Ҷ
	timer.start();
	vector<FanBlade> fan_boxes;
	fan_boxes = getFanBlade(mask);
	if (!fan_boxes.empty())
	{
		//cout << "�м�⵽��Ҷ" << endl;
		showFan(fan_boxes, img);
		lighted_fanblade = fan_boxes;
	}
	else
	{
		//cout << "δ��⵽��Ҷ" << endl;
	}
	timer.printElapsed("�����Ҷ");

	// 5.����Ŀ����Ҷ���ж���ת����
	timer.start();
	trackFanBlade(); // ����Ŀ����Ҷ
	if (is_tracking) // ����и���Ŀ�꣬��ʼ�жϲ�������ת����
	{
		updateRotationDirection();
	}

	if (rotation_direction == RotationDirection::ANTICLOCKWISE)
	{
		cout << "��ʱ����ת" << endl;
	}
	else if (rotation_direction == RotationDirection::CLOCKWISE)
	{
		cout << "˳ʱ����ת" << endl;
	}
	else
	{
		cout << "δ֪" << endl;
	}
	timer.printElapsed("������Ҷ���ж���ת����");
	imshow("result", img);
	return r_success;
}

void BuffTracker::getMask(const Mat& src, Mat& mask)
{
	// 1.ʹ��ͨ�������������ɫ��ȡ
	Mat binary;
	vector<Mat> channels;
	split(src, channels);

	//����ʶ���װ�װ���ɫ��Ϊ��ɫ
	Mat blueChannel = channels[0] - channels[2];

	// 2.��ֵ������
	threshold(blueChannel, binary, Config::BINARY_THRESHHOLD, 255, THRESH_BINARY);

	// 3.��̬ѧ����ȥ��
	Mat kernel1 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(binary, binary, MORPH_CLOSE, kernel1);

	// 4.���Ͳ���
	Mat kernel2 = getStructuringElement(MORPH_DILATE, Size(Config::morph_kernel_size, Config::morph_kernel_size));
	dilate(binary, mask, kernel2, Point(-1, -1), Config::morph_iterations);

	imshow("", mask);
}

void BuffTracker::getAlternateBoxes(const Mat& mask, vector<BBox>& alternateboxes, Mat& img)
{
	vector<vector<Point>> contours;
	// 1.Ѱ������
	findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	drawContours(img, contours, -1, Scalar(255, 0, 0));
	// 2.���������ҵ�������
	for (const auto& contour : contours)
	{
		// 3.��ȡ��������С��Ӿ���
		Rect rect = boundingRect(contour);
		// 4.���ɸѡ
		double area = contourArea(contour);
		if (area < Config::r_center_min_area || area > Config::r_center_max_area)
		{
			//cout << "û��ͨ�����ɸѡ" << endl;
			continue;
		}
		// 5.�����ɸѡ
		float aspec_ratio = max(rect.width, rect.height) / min(rect.width, rect.height);
		if (aspec_ratio < Config::r_center_min_ratio || aspec_ratio > Config::r_center_max_ratio)
		{
			//cout << "û��ͨ�������ɸѡ" << endl;
			continue;
		}
		// 6.��ǰ֡����R��������һ֡����R���ĵľ���ɸѡ�������ȥʶ��Ч�����ȶ���
		//Point2f current_center = Point2f(rect.x + rect.width / 2.0f, rect.y + rect.height / 2.0f);
		//double distance = norm(current_center - last_center);
		//if (distance > 3 * radius)
		//{
		//	cout << "û��ͨ������ɸѡ" << endl;
		//	continue;
		//}
		BBox box(rect, 0); // ��Rect����ת��ΪBBox���͵�
		alternateboxes.push_back(box);
	}
}
bool BuffTracker::compareByCIOU(const vector<BBox>& alternateboxes, BBox& bestbox)
{
	if (alternateboxes.empty())
	{
		//cout << "û�к�ѡ��ƥ��ʧ��" << endl;
		return false; // û�к�ѡ��ƥ��ʧ��
	}
	float best_ciou = -1.0f;
	float bestindex = -1;
	// 1.�������к�ѡ�򣬼�������һ֡����R��CIOU
	for (size_t i = 0;i < alternateboxes.size();i++)
	{
		float ciou = alternateboxes[i].ciou(r_box);
		if (ciou > best_ciou)
		{
			bestindex = i;
			best_ciou = ciou;
		}
	}
	// 2.����ҵ������ź�ѡ��
	if (bestindex != -1)
	{
		bestbox = alternateboxes[bestindex];
		return true;
	}
	//cout << "û���ҵ����ź�ѡ��" << endl;
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

	// 1.Ѱ������
	vector<vector<Point>> contours;
	findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	for (const auto& contour : contours)
	{
		// 2.���ɸѡ
		double area = contourArea(contour);
		if (area < Config::fan_blade_min_area || area > Config::fan_blade_max_area)
		{
			//cout << "��Ҷûͨ�����ɸѡ" << endl;
			continue;
		}
		// 3.��ȡ��ת����ɸѡ
		RotatedRect fan_rrect = minAreaRect(contour);
		// 4.�����ɸѡ��������ת������Ϊ��Ҷ����ת���壬����ת��������ʣ�
		float w = fan_rrect.size.width;
		float h = fan_rrect.size.height;
		float aspect_ratio = max(w, h) / min(w, h);
		if (aspect_ratio < Config::fan_blade_min_ratio || aspect_ratio > Config::fan_blade_max_ratio)
		{
			//cout << "��Ҷδͨ�������ɸѡ" << endl;
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
		// ��ȡ��Ҷ����ת����
		RotatedRect f_rrect = f_box.box; 
		// ��ȡ��ת���ε��ĸ�����
		Point2f vertices[4];
		f_rrect.points(vertices);
		// ѭ������������
		for (int i = 0; i < 4; i++)
		{
			line(img, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 255), 2);
		}
	}
}
void BuffTracker::trackFanBlade()
{
	if (lighted_fanblade.empty()) // ���û�м�⵽��������Ҷ
	{
		is_tracking = false;
		return;
	}
	if (!is_tracking) // ���֮ǰû��Ŀ����Ҷ������ѡ���Ĵ�������Ҷ�е�һ��
	{
		target_fanblade = lighted_fanblade[0];
		is_tracking = true;
	}
	else // ����Ѿ���Ŀ����Ҷ��Ѱ���������Ҷ��Ϊ��ǰ֡��׷����Ҷ
	{
		float min_dist = -1.0f;
		int best_indest = -1;
		for (size_t i = 0; i < lighted_fanblade.size(); i++)
		{
			// ��������һ֡Ŀ����Ҷ�ľ���
			Point2f diff = lighted_fanblade[i].center - target_fanblade.center;
			float distance = norm(diff);
			if (best_indest == -1 || distance < min_dist)
			{
				min_dist = distance;
				best_indest = i;
			}
		}
		// ����Ŀ��
		target_fanblade = lighted_fanblade[best_indest];
	}
}
void BuffTracker::updateRotationDirection()
{
	// 1.���㵱ǰ��ԭʼ�Ƕȣ����������R��
	float current_raw_angle = atan2((target_fanblade.center.y - r_box.center().y), target_fanblade.center.x - r_box.center().x); 
	// ����ǵ�һ�Σ���ʼ���Ƕ�
	if (!angle_diff_history.empty())
	{
		last_raw_angle = current_raw_angle;
		continuous_angle = current_raw_angle;
		angle_diff_history.push_back(0.0f);
	}
	// 2.�ǶȽ��㣨����-179�㵽1����������ĽǶ�ͻ�䣩
	float angle_diff = current_raw_angle - last_raw_angle; // ������֮֡��ĽǶȱ仯
	if (angle_diff > CV_PI)
	{
		angle_diff -= 2 * CV_PI;
	}
	else if (angle_diff < -CV_PI)
	{
		angle_diff += 2 * CV_PI;
	}
	// ���������ǶȺ���һ֡�Ƕ�
	continuous_angle += angle_diff;
	last_raw_angle = current_raw_angle;
	// 3.�����εĽǶȱ仯��������ʷ����
	angle_diff_history.push_back(angle_diff);
	if (angle_diff_history.size() > 10)
	{
		angle_diff_history.pop_front(); // ���ֶ��д�С
	}
	// ������ʷ���������нǶȱ仯���ܺ�
	float total_diff = 0.0f;
	for (const auto& diff : angle_diff_history)
	{
		total_diff += diff;
	}
	// �����ܺ͵ķ����жϷ���,����һ��С����ֵ������������
	if (total_diff > 0.01)
	{
		rotation_direction = RotationDirection::ANTICLOCKWISE; // ��Ϊ��ʱ�뷽��
	}
	else if (total_diff < -0.01)
	{
		rotation_direction = RotationDirection::CLOCKWISE; // ��Ϊ˳ʱ�뷽��
	}
	else
	{
		rotation_direction = RotationDirection::UNKONW; // ��Ϊδ֪
	}
}