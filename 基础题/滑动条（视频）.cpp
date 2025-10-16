/*
˼·��������ͷ�޷������ع⣩��
��1����main�����������ںͻ�������
��2�������������������������Ӧ�������ع�ʱ������ȣ�������brightness��exposureΪ�������Ĳ�����
	�ڸ��Ե���Ӧ��������actual_brightness��actual_exposureӳ�䵽���ʵķ�Χ��ʹ��set()����ȥ�����ع�ʱ������ȣ�
��3��Ȼ����main��������VideoWriter����������Ƶ¼�ƣ�����bool���͵�recording��ʾ�Ƿ�¼�� false��ʾֹͣ¼�� true��ʾ��ʼ¼��
��4����whileѭ������write()������ͷ��ͼ��д����Ƶ֡���ÿո����¼�ƵĿ�ʼ��ֹͣ����Esc���˳���
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
int brightness = 50;    // ����ֵ (0-100)
int exposure = 50;      // �ع�ֵ (0-100)
// ���Ȼص�����
void onBrightnessChange(int value, void* userdata) {
	// ��������ֵӳ�䵽ʵ�����ȷ�Χ (-64 �� 64)
	int actual_brightness = value;
	capture.set(CAP_PROP_BRIGHTNESS, actual_brightness);
	cout << "��������Ϊ: " << actual_brightness << endl;
}

// �ع�ص�����
void onExposureChange(int value, void* userdata) {
	// ��������ֵӳ�䵽�عⷶΧ (-7 �� -1����ֵ��ʾ�Զ��ع�ر�)
	double actual_exposure = value / 10.0 - 5;
	capture.set(CAP_PROP_EXPOSURE, actual_exposure);
	cout << "�ع�����Ϊ: " << actual_exposure << endl;
}


int main()
{
	system("color F0"); //�������������ɫ

	capture.open(0);
	if (!capture.read(frame))
	{
		cout << "����ͷ����" << endl;
		return -1;
	}

	// ��������
	namedWindow("����ͷ", WINDOW_AUTOSIZE);

	// ����������
	createTrackbar("����", "����ͷ", NULL, 100, onBrightnessChange);
	createTrackbar("�ع�", "����ͷ", NULL, 100, onExposureChange);

	double fps = capture.get(CAP_PROP_FPS);
	double width = capture.get(CAP_PROP_FRAME_WIDTH);
	double height = capture.get(CAP_PROP_FRAME_HEIGHT);

	string fps_str = "FPS:" + to_string((int)fps);
	string width_str = "Width:" + to_string((int)width);
	string height_str = "Height:" + to_string((int)height);

	cout << "��Ƶͼ��֡�ʣ�" << fps << endl;
	cout << "��Ƶͼ���ȣ�" << width << endl;
	cout << "��Ƶͼ��߶ȣ�" << height << endl;

	// ����VideoWriter����
	string filename = "D:\\Opencvpracticedata\\output.avi";
	int fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G'); // �����ʽ
	VideoWriter writer(filename, fourcc, fps, Size((int)width, (int)height));

	bool recording = false;
	cout << "���ո����ʼ/ֹͣ¼�ƣ���ESC�˳�" << endl;

	while (true)
	{
		if (!capture.read(frame))
		{
			cout << "����ͷ��ȡʧ��" << endl;
			break;
		}

		if (recording)
		{
			putText(frame, "RECODING", Point(10, 30), FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0), 2);
			writer.write(frame); // д����Ƶ֡
		}

		putText(frame, fps_str, Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
		putText(frame, width_str, Point(10, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
		putText(frame, height_str, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

		imshow("����ͷ", frame);

		char c = (char)waitKey(5);
		if (c == 27) break;
		if (c == 32) // �ո��л�¼��״̬
		{
			recording = !recording;
			cout << (recording ? "��ʼ¼��" : "¼�ƽ���") << endl;
		}
	}


	waitKey(0);
	return 0;
}

