#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "config.hpp"
#include "BuffTracker.hpp"

using namespace std;
using namespace cv;

namespace Config
{
	// Ԥ�������
	int BINARY_THRESHHOLD = 100; // ��ֵ����ֵ
	int morph_kernel_size = 3; // ���ͺ˴�С
	int morph_iterations = 4; // ���͵�������

	// ����Rʶ�����
	double r_center_min_area = 220.0; // ��С���
	double r_center_max_area = 1000.0; // ������
	double r_center_min_ratio = 0.5; // ��С�����
	double r_center_max_ratio = 1.2; // ��󳤿��

	// ��Ҷ������
	double fan_blade_min_area = 1500.0; // ��Ҷ��С���
	double fan_blade_max_area = 5000.0; // ��Ҷ������
	double fan_blade_min_ratio = 1.0; // ��Ҷ��С�����
	double fan_blade_max_ratio = 8.0; // ��Ҷ��󳤿��

	//float raduis = 57.93; // �������ذ뾶����Ҷ���ĵ�����R�ľ��룩����Ҫ����
};


int main()
{

	system("color F0");
	string video_path = "D:\\Opencvpracticedata\\blue.mp4";
	Mat srcImg;

	VideoCapture cap(video_path);
	if (!cap.read(srcImg))
	{
		cout << "��Ƶ����" << endl;
		return -1;
	}
	long framecount = 0; // ��¼֡��
	int startframe = 30; // Ҫ����selectROIʱ��֡��
	unique_ptr<BuffTracker> tracker = nullptr; //��������׷����
	Rect r_rect(0,0,0,0); // ����R��ROI
	Rect fan_rect(0,0,0,0); // ��Ҷ��ROI

	int frameCount = 0; // ��ȡ��֡��
	SimpleTimer fpsTimer; // ���ڼ���FPS�ļ�ʱ��
	fpsTimer.start();
	double fps = 0.0f; // �洢����õ���FPS
	string str = ""; // ������ʾ���ַ���

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
		// ÿ�����һ��FPS
		if (elapsed >= 1000.0f)
		{
			fps = frameCount / (elapsed / 1000.0f);
			frameCount = 0;
			fpsTimer.start();
			str = "FPS:" + to_string(fps); // ����ʵʱ��ʾFPS
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