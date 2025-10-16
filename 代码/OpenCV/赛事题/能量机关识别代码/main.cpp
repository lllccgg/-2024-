#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "config.hpp"
#include "BuffTracker.hpp"

using namespace std;
using namespace cv;

namespace Config
{
	// 预处理参数
	int BINARY_THRESHHOLD = 100; // 二值化阈值
	int morph_kernel_size = 3; // 膨胀核大小
	int morph_iterations = 4; // 膨胀迭代次数

	// 中心R识别参数
	double r_center_min_area = 220.0; // 最小面积
	double r_center_max_area = 1000.0; // 最大面积
	double r_center_min_ratio = 0.5; // 最小长宽比
	double r_center_max_ratio = 1.2; // 最大长宽比

	// 扇叶检测参数
	double fan_blade_min_area = 1500.0; // 扇叶最小面积
	double fan_blade_max_area = 5000.0; // 扇叶最大面积
	double fan_blade_min_ratio = 1.0; // 扇叶最小长宽比
	double fan_blade_max_ratio = 8.0; // 扇叶最大长宽比

	//float raduis = 57.93; // 能量机关半径（扇叶中心到中心R的距离），需要调整
};


int main()
{

	system("color F0");
	string video_path = "D:\\Opencvpracticedata\\blue.mp4";
	Mat srcImg;

	VideoCapture cap(video_path);
	if (!cap.read(srcImg))
	{
		cout << "视频有误！" << endl;
		return -1;
	}
	long framecount = 0; // 记录帧数
	int startframe = 30; // 要进行selectROI时的帧数
	unique_ptr<BuffTracker> tracker = nullptr; //能量机关追踪器
	Rect r_rect(0,0,0,0); // 中心R的ROI
	Rect fan_rect(0,0,0,0); // 扇叶的ROI

	int frameCount = 0; // 获取的帧数
	SimpleTimer fpsTimer; // 用于计算FPS的计时器
	fpsTimer.start();
	double fps = 0.0f; // 存储计算得到的FPS
	string str = ""; // 用于显示的字符串

	while (true)
	{
		if (!cap.read(srcImg))
		{
			break;
		}
		Mat debugImg;
		srcImg.copyTo(debugImg);

		frameCount++;
		double elapsed = fpsTimer.getElapsedMs();
		// 每秒计算一次FPS
		if (elapsed >= 1000.0f)
		{
			fps = frameCount / (elapsed / 1000.0f);
			frameCount = 0;
			fpsTimer.start();
			str = "FPS:" + to_string(fps); // 用于实时显示FPS
		}
		putText(debugImg, str, Point2f(100, 100), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0));

		if (framecount == startframe)
		{
			r_rect = selectROI("R", srcImg.clone());
			fan_rect = selectROI("Fan", srcImg.clone());
		}
		if (r_rect.width > 0 && fan_rect.width > 0 && framecount == startframe)
		{
			BBox r_box = BBox(r_rect, 0);
			BBox fan_box = BBox(fan_rect, 1);

			tracker = make_unique<BuffTracker>(r_box, fan_box);
		}
		if (tracker)
		{
			bool success = tracker->update(debugImg);
		}
		framecount++;


		namedWindow("src", WINDOW_GUI_NORMAL);
		imshow("src", srcImg);

		if ((char)waitKey(1) == 27)
		{
			break;
		}

	}

	waitKey(0);
	return 0;
}