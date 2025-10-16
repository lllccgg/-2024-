/*
思路（本摄像头无法调节曝光）：
（1）在main函数创建窗口和滑动条；
（2）定义和声明两个滑动条的响应函数（曝光时间和亮度），定义brightness、exposure为滑动条的参数，
	在各自的响应函数定义actual_brightness和actual_exposure映射到合适的范围，使用set()函数去设置曝光时间和亮度，
（3）然后在main函数创建VideoWriter对象，用于视频录制，定义bool类型的recording表示是否录制 false表示停止录制 true表示开始录制
（4）在while循环中用write()将摄像头的图像写入视频帧，用空格控制录制的开始和停止，用Esc来退出。
*/
#include <opencv2\opencv.hpp>
#include <opencv2\ximgproc.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>

using namespace std;
using namespace cv;

Mat frame,img1;

VideoCapture capture;
int brightness = 50;    // 亮度值 (0-100)
int exposure = 50;      // 曝光值 (0-100)
// 亮度回调函数
void onBrightnessChange(int value, void* userdata) {
	// 将滑动条值映射到实际亮度范围 (-64 到 64)
	int actual_brightness = value;
	capture.set(CAP_PROP_BRIGHTNESS, actual_brightness);
	cout << "亮度设置为: " << actual_brightness << endl;
}

// 曝光回调函数
void onExposureChange(int value, void* userdata) {
	// 将滑动条值映射到曝光范围 (-7 到 -1，负值表示自动曝光关闭)
	double actual_exposure = value / 10.0 - 5;
	capture.set(CAP_PROP_EXPOSURE, actual_exposure);
	cout << "曝光设置为: " << actual_exposure << endl;
}


int main()
{
	system("color F0"); //更改输出界面颜色

	capture.open(0);
	if (!capture.read(frame))
	{
		cout << "摄像头有误" << endl;
		return -1;
	}

	// 创建窗口
	namedWindow("摄像头", WINDOW_AUTOSIZE);

	// 创建滑动条
	createTrackbar("亮度", "摄像头", NULL, 100, onBrightnessChange);
	createTrackbar("曝光", "摄像头", NULL, 100, onExposureChange);

	double fps = capture.get(CAP_PROP_FPS);
	double width = capture.get(CAP_PROP_FRAME_WIDTH);
	double height = capture.get(CAP_PROP_FRAME_HEIGHT);

	string fps_str = "FPS:" + to_string((int)fps);
	string width_str = "Width:" + to_string((int)width);
	string height_str = "Height:" + to_string((int)height);

	cout << "视频图像帧率：" << fps << endl;
	cout << "视频图像宽度：" << width << endl;
	cout << "视频图像高度：" << height << endl;

	// 创建VideoWriter对象
	string filename = "D:\\Opencvpracticedata\\output.avi";
	int fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G'); // 编码格式
	VideoWriter writer(filename, fourcc, fps, Size((int)width, (int)height));

	bool recording = false;
	cout << "按空格键开始/停止录制，按ESC退出" << endl;

	while (true)
	{
		if (!capture.read(frame))
		{
			cout << "摄像头读取失败" << endl;
			break;
		}

		if (recording)
		{
			putText(frame, "RECODING", Point(10, 30), FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0), 2);
			writer.write(frame); // 写入视频帧
		}

		putText(frame, fps_str, Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
		putText(frame, width_str, Point(10, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
		putText(frame, height_str, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

		imshow("摄像头", frame);

		char c = (char)waitKey(5);
		if (c == 27) break;
		if (c == 32) // 空格切换录制状态
		{
			recording = !recording;
			cout << (recording ? "开始录制" : "录制结束") << endl;
		}
	}


	waitKey(0);
	return 0;
}

