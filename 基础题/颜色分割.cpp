/*
˼·��
��1����BGRת��ΪHSV����ΪHSV�Թ��ո������У����ʺ���ɫ�ָ
��2�����ں�ɫ��������HSV����(0,50,50)-(10,255,255)��(170,50,50)-(180,255,255)�����Խ�����������Χ��
��3����ͨ��inRange()������img��������ɫ��Χ��ȡ������������Ƕ�ֵ������ͼ��mask1��mask2��
��4����mask1��mask2ȡ�������ָ�������ĺ�ɫͼ��mask����ʱmaskͼ���ǳ��˺�ɫ����Ϊ�ף�255����������Ϊ�ڣ�0����
��5����ˣ�����Canny��ȡ����ɫͼ��ı�Եcanny��
��6�������CV_8U��ͨ��ͼ��canny_color���ڸ�ͼ���л�����ɫͼ�εı�Ե��ʹ��setTo()����������Եͼ����canny�ķ��㲿����Ϊǳ��ɫ��
��7��ͨ������������S1��S2��V1��V2��ֵ�����ﵽ���õطָ��ɫͼ���Ŀ�ġ�����ȷ����S1=170��S2=135��V1=175��V2=156��
*/


#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;

int s1 = 170; // S1�ĳ�ʼֵ
int s2 = 135; // S2�ĳ�ʼֵ
int v1 = 175; // V1�ĳ�ʼֵ
int v2 = 156; // V2�ĳ�ʼֵ

void s1_trackbarcallback(int value, void* userdata)
{
	s1 = value;
}
void s2_trackbarcallback(int value, void* userdata)
{
	s2 = value;
}
void v1_trackbarcallback(int value, void* userdata)
{
	v1 = value;
}
void v2_trackbarcallback(int value, void* userdata)
{
	v2 = value;
}

Mat img;

int main()
{
	system("color F0"); //�������������ɫ

	img = imread("D:\\Opencvpracticedata\\0.png");
	Mat hsv;
	cvtColor(img, hsv, COLOR_BGR2HSV);

	
	namedWindow("������", WINDOW_GUI_NORMAL);


	createTrackbar("S1_min", "������", NULL, 255, s1_trackbarcallback);
	createTrackbar("S2_min", "������", NULL, 255, s2_trackbarcallback);
	createTrackbar("V1_min", "������", NULL, 255, v1_trackbarcallback);
	createTrackbar("V2_min", "������", NULL, 255, v2_trackbarcallback);

	while (true)
	{
		Scalar lower_r1(0, s1, v1);
		Scalar upper_r1(10, 255, 255);
		Scalar lower_r2(170, s2, v2);
		Scalar upper_r2(180, 255, 255);
		
		Mat mask1, mask2, mask;
		inRange(hsv, lower_r1, upper_r1, mask1); // mask1��0-10�ȵĺ�ɫ����,������Ƕ�ֵ������ͼ��
		inRange(hsv, lower_r2, upper_r2, mask2); // mask2��170-180�ȵĺ�ɫ���֣�������Ƕ�ֵ������ͼ��
		bitwise_or(mask1, mask2, mask); // mask�����յĺ�ɫ���֣��ָ����ɫ���֣�

		Mat canny = Mat::zeros(img.size(), CV_8UC1);
		Canny(mask, canny, 245, 255);
	
		Mat canny_color = Mat::zeros(img.size(), CV_8UC3);
		canny_color.setTo(Scalar(255, 255, 0), canny); // ����Եͼ��canny�еķ��㲿����Ϊǳ��ɫ



		//namedWindow("ԭͼ", WINDOW_AUTOSIZE);
		//namedWindow("��ɫ�ָ�󣨶�ֵ����", WINDOW_GUI_NORMAL);
		//namedWindow("��Ե����", WINDOW_GUI_NORMAL);
		//namedWindow("��Ե����_��ɫ", WINDOW_GUI_NORMAL);
		imshow("ԭͼ", img);
		imshow("��ɫ�ָ�󣨶�ֵ����", mask);
		imshow("��Ե����", canny);
		imshow("��Ե����_��ɫ", canny_color);


		if ((char)waitKey(1) == 27)
		{
			imwrite("D:\\Opencvpracticedata\\��Եͼ��.png", canny_color); // �����Եͼ��
			break;
		}

	}

	return 0;
}




