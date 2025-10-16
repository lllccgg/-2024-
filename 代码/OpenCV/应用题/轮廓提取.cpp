/*
˼·��
��1����ͼ��ת��ΪHSV����������ɫ�ָ�Ա����������ȡ��
��2������ƻ���Ǻ�ɫ�ͻ�ɫ�����Դ���mask1��mask2��mask3���ָ����ɫ�ͻ�ɫ��Χ�Ķ�ֵ������ͼ��mask��
��3��ʹ�ÿ�����ȥ����С����㣨������֦����Ҷ�Ķ�ֵ��ͼ�񣩣�
��4��ʹ��findContours��ȡ��������
��5��ʹ��contourArea�����õ�������������ȥ��С�������������֦����Ҷ�ȵ��������������������һ����������ƻ���ģ�
��6��ʹ��boundingRect��ԭͼ�ϻ���ƻ������Ӿ�������סƻ����
��7������һ������Ϊ�ڵ�ͼ��ʹ��drawContours����ƻ����������ɫΪ��ɫ��
*/

#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;

int s1 = 123;
int s2 = 75;
int s3 = 144;
int v1 = 75;
int v2 = 43;
int v3 = 101;
int h3 = 3;
int h4 = 25;

void s1_trackbarcallback(int value, void* userdata)
{
	s1 = value;
}
void s2_trackbarcallbcak(int value, void* userdata)
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
void s3_trackbarcallback(int value, void* userdata)
{
	s3 = value;
}
void v3_trackbarcallback(int value, void* userdata)
{
	v3 = value;
}
void h3_trackbarcallback(int value, void* userdata)
{
	h3 = value;
}
void h4_trackbarcallback(int value, void* userdata)
{
	h4 = value;
}

Mat img;
const string IMAGE_PATH = "D:\\Opencvpracticedata\\apple.png"; // ��ȡ·��

int main()
{
	system("color F0"); //�������������ɫ

	img = imread(IMAGE_PATH);
	if (img.empty())
	{
		cout << "ͼƬ����" << endl;
		return -1;
	}

	namedWindow("������", WINDOW_GUI_NORMAL);
	createTrackbar("s1", "������", NULL, 255, s1_trackbarcallback);
	createTrackbar("s2", "������", NULL, 255, s2_trackbarcallbcak);
	createTrackbar("v1", "������", NULL, 255, v1_trackbarcallback);
	createTrackbar("v2", "������", NULL, 255, v2_trackbarcallback);
	createTrackbar("s3", "������", NULL, 255, s3_trackbarcallback);
	createTrackbar("v3", "������", NULL, 255, v3_trackbarcallback);
	createTrackbar("h3", "������", NULL, 255, h3_trackbarcallback);
	createTrackbar("h4", "������", NULL, 255, h4_trackbarcallback);

	Mat hsv;
	cvtColor(img, hsv, COLOR_BGR2HSV); // ת��ΪHSV��ɫ�ռ�
	Mat mask1, mask2, mask3, mask;

	while (true)
	{
		Mat debugimg = img.clone();

		Scalar lower_r1(0, s1, v1);
		Scalar upper_r1(10, 255, 255);
		Scalar lower_r2(160, s2, v2);
		Scalar upper_r2(180, 255, 255);
		Scalar lower_r3(h3,s3,v3);
		Scalar upper_r4(h4,255,255);
		inRange(hsv, lower_r1, upper_r1, mask1); // mask1��0-10�ȵĺ�ɫ����,������Ƕ�ֵ������ͼ��
		inRange(hsv, lower_r2, upper_r2, mask2); // mask2��170-180�ȵĺ�ɫ���֣�������Ƕ�ֵ������ͼ��
		inRange(hsv, lower_r3, upper_r4, mask3); 
		bitwise_or(mask1, mask2, mask); // mask�����յĺ�ɫ���֣��ָ����ɫ���֣�
		bitwise_or(mask3, mask, mask); // �����ƻ���еĻ�ɫ����


		morphologyEx(mask, mask, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(13, 13))); // �����㣬ȥ�����
		imshow("�������", mask);

		vector<vector<Point>> contours; // ���ڴ洢�����������
		findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); // ֻ���������
		vector<vector<Point>> valid_contours;
		for (size_t i = 0; i < contours.size(); i++)
		{
			double area = contourArea(contours[i]); // �����������
			if (area > 10000) // ɸ����Ҷ����֦������
			{
				valid_contours.push_back(contours[i]);

			}
		}

		cout << "��Ч��������: " << valid_contours.size() << endl;

		for (size_t i = 0; i < valid_contours.size(); i++)
		{
			Rect rect = boundingRect(valid_contours[i]);
			rectangle(debugimg, rect, Scalar(0, 255, 0), 2); // ��ԭͼ�ϻ��Ʊ߽��
		}

	
		imshow("�����", debugimg);

		Mat save_img = Mat::zeros(img.size(),img.type());
		drawContours(save_img, valid_contours, -1, Scalar(255, 0, 0), 3); // �ڱ���Ϊ�ڵ�ͼ�ϻ���������Ч����
		imshow("������Ч����", save_img);

		if ((char)waitKey(1) == 27)
		{
			break;
		}

	}


	return 0;
}



