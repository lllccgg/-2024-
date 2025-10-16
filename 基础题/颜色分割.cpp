/*
思路：
（1）将BGR转化为HSV，因为HSV对光照更不敏感，更适合颜色分割；
（2）由于红色具有两个HSV，即(0,50,50)-(10,255,255)、(170,50,50)-(180,255,255)，所以将设置两个范围；
（3）再通过inRange()函数将img的两个红色范围提取出来，输出的是二值化掩码图像，mask1、mask2；
（4）将mask1、mask2取交集，分割出完整的红色图像mask。此时mask图像是除了红色部分为白（255）、其他均为黑（0）；
（5）因此，采用Canny提取出红色图像的边缘canny；
（6）最后用CV_8U三通道图像canny_color，在该图像中画出红色图形的边缘，使用setTo()函数，将边缘图形中canny的非零部分设为浅蓝色。
（7）通过滑动条调节S1、S2、V1、V2的值，来达到更好地分割红色图像的目的。最终确定是S1=170、S2=135、V1=175、V2=156。
*/


#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;

int s1 = 170; // S1的初始值
int s2 = 135; // S2的初始值
int v1 = 175; // V1的初始值
int v2 = 156; // V2的初始值

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
	system("color F0"); //更改输出界面颜色

	img = imread("D:\\Opencvpracticedata\\0.png");
	Mat hsv;
	cvtColor(img, hsv, COLOR_BGR2HSV);

	
	namedWindow("滑动条", WINDOW_GUI_NORMAL);


	createTrackbar("S1_min", "滑动条", NULL, 255, s1_trackbarcallback);
	createTrackbar("S2_min", "滑动条", NULL, 255, s2_trackbarcallback);
	createTrackbar("V1_min", "滑动条", NULL, 255, v1_trackbarcallback);
	createTrackbar("V2_min", "滑动条", NULL, 255, v2_trackbarcallback);

	while (true)
	{
		Scalar lower_r1(0, s1, v1);
		Scalar upper_r1(10, 255, 255);
		Scalar lower_r2(170, s2, v2);
		Scalar upper_r2(180, 255, 255);
		
		Mat mask1, mask2, mask;
		inRange(hsv, lower_r1, upper_r1, mask1); // mask1是0-10度的红色部分,输出的是二值化掩码图像
		inRange(hsv, lower_r2, upper_r2, mask2); // mask2是170-180度的红色部分，输出的是二值化掩码图像
		bitwise_or(mask1, mask2, mask); // mask是最终的红色部分（分割出红色部分）

		Mat canny = Mat::zeros(img.size(), CV_8UC1);
		Canny(mask, canny, 245, 255);
	
		Mat canny_color = Mat::zeros(img.size(), CV_8UC3);
		canny_color.setTo(Scalar(255, 255, 0), canny); // 将边缘图像canny中的非零部分设为浅蓝色



		//namedWindow("原图", WINDOW_AUTOSIZE);
		//namedWindow("颜色分割后（二值化）", WINDOW_GUI_NORMAL);
		//namedWindow("边缘检测后", WINDOW_GUI_NORMAL);
		//namedWindow("边缘检测后_彩色", WINDOW_GUI_NORMAL);
		imshow("原图", img);
		imshow("颜色分割后（二值化）", mask);
		imshow("边缘检测后", canny);
		imshow("边缘检测后_彩色", canny_color);


		if ((char)waitKey(1) == 27)
		{
			imwrite("D:\\Opencvpracticedata\\边缘图像.png", canny_color); // 保存边缘图像
			break;
		}

	}

	return 0;
}




