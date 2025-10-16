/*
思路：
（1）在main函数里创建窗口和鼠标响应；
（2）鼠标响应：先定义一个变量end，0表示鼠标还没拖拽结束，1表示鼠标拖拽结束；
				当鼠标左键按下时，为初始点；
				当鼠标左键未松开，且拖拽时，为过程点，画出矩形（初始点为矩形左上角顶点，过程点为当前矩形右下角顶点，会一直更新），
				为了避免每次拖拽会累计矩形的生成，会在这种情况开头从img深拷贝出新图像来画出矩形。拖拽期间，会输出经过点的坐标和BGR值；
				当鼠标松开时，令end = 1，表示拖拽结束，且记录结束点；
				当end = 1时，计算出最终矩形的中心坐标（用初始点和结束点坐标相加再除以2），并画出最终矩形，
				且ROI以矩形的初始点和结束点中坐标的较小的为点，然后取两个坐标相减的绝对值作为长和宽，然后ROI与整个图像img的Rect取交集rect，确定框住的图像，达到边界检查的目的，
				最后显示框出的图像rect，再imwirte保存图像。

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

const string IMAGE_PATH = "D:\\Opencvpracticedata\\cat.png"; // 读取路径
const string SAVE_PATH = "D:\\Opencvpracticedata\\save_cat.png"; // 保存路径

int main()
{
	system("color F0"); //更改输出界面颜色

	img = imread(IMAGE_PATH);
	if (img.empty())
	{
		cout << "图片有误" << endl;
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
	int end_flag = 0; // 0-代表未结束 1-代表结束

	if (event == EVENT_LBUTTONDOWN)
	{
		prePoint = Point(x, y);
	}
	if (event == EVENT_MOUSEMOVE && (flags && EVENT_FLAG_LBUTTON))
	{
		imgTemp.copyTo(img);  // 关键：先恢复原始图像
		tempPoint = Point(x, y);
		rectangle(img, prePoint, tempPoint, Scalar(0, 0, 255), 2);

		int px = x;
		int py = y;
		if (px >= 0 && px < imgTemp.cols && py >= 0 && py < imgTemp.rows) // 边界检查
		{
			Scalar color = imgTemp.at<Vec3b>(py, px);
			cout << "当前鼠标位置坐标为(" << px << "," << py << ")"
				<< "  颜色值为(BGR):(" << (int)color[0] << "," << (int)color[1] << "," << (int)color[2] << ")" << endl;
		}
		imshow("img", img);
	}
	if (event == EVENT_LBUTTONUP)
	{
		endPoint = Point(x, y);
		end_flag = 1; // 拖拽结束
	}

	if (end_flag == 1)
	{
		int center_x = (prePoint.x + endPoint.x) / 2;
		int center_y = (prePoint.y + endPoint.y) / 2;
		cout << "框中心点坐标为(" << center_x << "," << center_y << ")" << endl;

		rectangle(img, prePoint, endPoint, Scalar(0, 255, 0), 2);

		Rect rect(min(prePoint.x, endPoint.x), min(prePoint.y, endPoint.y),abs(endPoint.x - prePoint.x), abs(endPoint.y - prePoint.y));
		rect &= Rect(0, 0, imgTemp.cols, imgTemp.rows); // 边界检查，确保不超出图像范围（两个矩形取交集）
		if (rect.width > 0 && rect.height > 0)
		{
			imgPoint = imgTemp(rect);
			imshow("imgPoint", imgPoint);
			imwrite(SAVE_PATH, imgPoint);
		}



	}

}

