/*
˼·��
��1����main�����ﴴ�����ں������Ӧ��
��2�������Ӧ���ȶ���һ������end��0��ʾ��껹û��ק������1��ʾ�����ק������
				������������ʱ��Ϊ��ʼ�㣻
				��������δ�ɿ�������קʱ��Ϊ���̵㣬�������Σ���ʼ��Ϊ�������ϽǶ��㣬���̵�Ϊ��ǰ�������½Ƕ��㣬��һֱ���£���
				Ϊ�˱���ÿ����ק���ۼƾ��ε����ɣ��������������ͷ��img�������ͼ�����������Ρ���ק�ڼ䣬�����������������BGRֵ��
				������ɿ�ʱ����end = 1����ʾ��ק�������Ҽ�¼�����㣻
				��end = 1ʱ����������վ��ε��������꣨�ó�ʼ��ͽ�������������ٳ���2�������������վ��Σ�
				��ROI�Ծ��εĳ�ʼ��ͽ�����������Ľ�С��Ϊ�㣬Ȼ��ȡ������������ľ���ֵ��Ϊ���Ϳ�Ȼ��ROI������ͼ��img��Rectȡ����rect��ȷ����ס��ͼ�񣬴ﵽ�߽����Ŀ�ģ�
				�����ʾ�����ͼ��rect����imwirte����ͼ��

*/


#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;

void mouse(int event,int x,int y,int flags,void*);

Mat img, imgPoint, imgTemp;
Point prePoint, endPoint, tempPoint;

const string IMAGE_PATH = "D:\\Opencvpracticedata\\cat.png"; // ��ȡ·��
const string SAVE_PATH = "D:\\Opencvpracticedata\\save_cat.png"; // ����·��

int main()
{
	system("color F0"); //�������������ɫ

	img = imread(IMAGE_PATH);
	if (img.empty())
	{
		cout << "ͼƬ����" << endl;
		return -1;
	}

	img.copyTo(imgTemp);

	namedWindow("img",WINDOW_GUI_NORMAL);
	imshow("img", img);
	setMouseCallback("img", mouse);
	

	waitKey(0);
	return 0;
}

void mouse(int event, int x, int y, int flags, void*)
{
	int end_flag = 0; // 0-����δ���� 1-�������

	if (event == EVENT_LBUTTONDOWN)
	{
		prePoint = Point(x, y);
	}
	if (event == EVENT_MOUSEMOVE && (flags && EVENT_FLAG_LBUTTON))
	{
		imgTemp.copyTo(img);  // �ؼ����Ȼָ�ԭʼͼ��
		tempPoint = Point(x, y);
		rectangle(img, prePoint, tempPoint, Scalar(0, 0, 255), 2);

		int px = x;
		int py = y;
		if (px >= 0 && px < imgTemp.cols && py >= 0 && py < imgTemp.rows) // �߽���
		{
			Scalar color = imgTemp.at<Vec3b>(py, px);
			cout << "��ǰ���λ������Ϊ(" << px << "," << py << ")"
				<< "  ��ɫֵΪ(BGR):(" << (int)color[0] << "," << (int)color[1] << "," << (int)color[2] << ")" << endl;
		}
		imshow("img", img);
	}
	if (event == EVENT_LBUTTONUP)
	{
		endPoint = Point(x, y);
		end_flag = 1; // ��ק����
	}

	if (end_flag == 1)
	{
		int center_x = (prePoint.x + endPoint.x) / 2;
		int center_y = (prePoint.y + endPoint.y) / 2;
		cout << "�����ĵ�����Ϊ(" << center_x << "," << center_y << ")" << endl;

		rectangle(img, prePoint, endPoint, Scalar(0, 255, 0), 2);

		Rect rect(min(prePoint.x, endPoint.x), min(prePoint.y, endPoint.y),abs(endPoint.x - prePoint.x), abs(endPoint.y - prePoint.y));
		rect &= Rect(0, 0, imgTemp.cols, imgTemp.rows); // �߽��飬ȷ��������ͼ��Χ����������ȡ������
		if (rect.width > 0 && rect.height > 0)
		{
			imgPoint = imgTemp(rect);
			imshow("imgPoint", imgPoint);
			imwrite(SAVE_PATH, imgPoint);
		}



	}

}

