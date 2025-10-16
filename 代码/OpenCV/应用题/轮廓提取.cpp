/*
思路：
（1）将图像转化为HSV，更容易颜色分割，以便进行轮廓提取；
（2）由于苹果是红色和黄色，所以创建mask1、mask2、mask3来分割出红色和黄色范围的二值化掩码图像mask；
（3）使用开运算去除较小的噪点（部分树枝和树叶的二值化图像）；
（4）使用findContours获取外轮廓；
（5）使用contourArea计算获得到的外轮廓，并去除小面积的轮廓（树枝和树叶等的轮廓），最后保留下来的一个轮廓就是苹果的；
（6）使用boundingRect在原图上画出苹果的外接矩形来框住苹果；
（7）创建一个背景为黑的图像，使用drawContours画出苹果轮廓，颜色为蓝色。
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
const string IMAGE_PATH = "D:\\Opencvpracticedata\\apple.png"; // 读取路径

int main()
{
	system("color F0"); //更改输出界面颜色

	img = imread(IMAGE_PATH);
	if (img.empty())
	{
		cout << "图片有误" << endl;
		return -1;
	}

	namedWindow("滑动条", WINDOW_GUI_NORMAL);
	createTrackbar("s1", "滑动条", NULL, 255, s1_trackbarcallback);
	createTrackbar("s2", "滑动条", NULL, 255, s2_trackbarcallbcak);
	createTrackbar("v1", "滑动条", NULL, 255, v1_trackbarcallback);
	createTrackbar("v2", "滑动条", NULL, 255, v2_trackbarcallback);
	createTrackbar("s3", "滑动条", NULL, 255, s3_trackbarcallback);
	createTrackbar("v3", "滑动条", NULL, 255, v3_trackbarcallback);
	createTrackbar("h3", "滑动条", NULL, 255, h3_trackbarcallback);
	createTrackbar("h4", "滑动条", NULL, 255, h4_trackbarcallback);

	Mat hsv;
	cvtColor(img, hsv, COLOR_BGR2HSV); // 转换为HSV颜色空间
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
		inRange(hsv, lower_r1, upper_r1, mask1); // mask1是0-10度的红色部分,输出的是二值化掩码图像
		inRange(hsv, lower_r2, upper_r2, mask2); // mask2是170-180度的红色部分，输出的是二值化掩码图像
		inRange(hsv, lower_r3, upper_r4, mask3); 
		bitwise_or(mask1, mask2, mask); // mask是最终的红色部分（分割出红色部分）
		bitwise_or(mask3, mask, mask); // 结合了苹果中的黄色部分


		morphologyEx(mask, mask, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(13, 13))); // 开运算，去除噪点
		imshow("开运算后", mask);

		vector<vector<Point>> contours; // 用于存储轮廓点的容器
		findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); // 只检测外轮廓
		vector<vector<Point>> valid_contours;
		for (size_t i = 0; i < contours.size(); i++)
		{
			double area = contourArea(contours[i]); // 计算轮廓面积
			if (area > 10000) // 筛掉树叶、树枝的轮廓
			{
				valid_contours.push_back(contours[i]);

			}
		}

		cout << "有效轮廓数量: " << valid_contours.size() << endl;

		for (size_t i = 0; i < valid_contours.size(); i++)
		{
			Rect rect = boundingRect(valid_contours[i]);
			rectangle(debugimg, rect, Scalar(0, 255, 0), 2); // 在原图上绘制边界框
		}

	
		imshow("检测结果", debugimg);

		Mat save_img = Mat::zeros(img.size(),img.type());
		drawContours(save_img, valid_contours, -1, Scalar(255, 0, 0), 3); // 在背景为黑的图上绘制所有有效轮廓
		imshow("所有有效轮廓", save_img);

		if ((char)waitKey(1) == 27)
		{
			break;
		}

	}


	return 0;
}



