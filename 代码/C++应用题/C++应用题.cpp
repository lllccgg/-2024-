#include <iostream>
#include <string>
#include <iomanip>
#include "Point.hpp"

using namespace std;

struct Rect 
{

public:
	int numID;
	int color;
	int width;
	int height;
	Point point; 
};

class Armor : public Rect //继承
{
public:
	Armor(const Rect& rect) : Rect(rect) {} // 初始化列表

	Point Central_Point(const Point point, int width, int height); // 计算出中心坐标
	double Diagonal(int width, int height); // 计算出对角线长度
	void Armor_Point(const Point point, int width, int height); // 输出四个顶点坐标
	void Armor_Color(int color); // 输出标定板颜色
};

// 计算出中心坐标
Point Armor::Central_Point(const Point point, int width, int height)
{
	Point center = Point(point.getX() + width / 2, point.getY() + height / 2);
	return center;
}

// 计算出对角线长度
double Armor::Diagonal(int width, int height)
{
	double diagonal = sqrt(width * width + height * height);
	return diagonal; 
}

// 输出四个顶点坐标
void Armor::Armor_Point(const Point point, int width, int height)
{
	Point left_top = point; // 左上
	Point right_top = Point(point.getX() + width, point.getY()); // 右上
	Point right_bottom = Point(point.getX() + width, point.getY() + height); // 右下
	Point left_bottom = Point(point.getX(), point.getY() + height); // 左下
	
	cout << "(" << left_top.getX() << "," << left_top.getY() << ")" << " "
		<< "(" << right_top.getX() << "," << right_top.getY() << ")" << " "
		<< "(" << right_bottom.getX() << "," << right_bottom.getY() << ")" << " "
		<< "(" << left_bottom.getX() << "," << left_bottom.getY() << ")" << endl;
}

// 输出标定板颜色
void Armor::Armor_Color(int color)
{
	if (color) // 1为红色
	{
		cout << "颜色：红" << endl;
	}
	else // 0为蓝色
	{
		cout << "颜色：蓝" << endl;
	}
}

int main()
{
	Rect rect;
	int numID, color;
	int px, py, width, height;
	cin >> numID >> color;
	cin >> px >> py >> width >> height;

	rect.numID = numID;
	rect.color = color;
	rect.point = Point(px, py);
	rect.width = width;
	rect.height = height;
	Armor armor(rect);

	cout << "ID：" << rect.numID << " "; 
	armor.Armor_Color(armor.color);

	Point center = armor.Central_Point(armor.point, armor.width, armor.height); // 中心坐标
	double diagonal = armor.Diagonal(armor.width, armor.height); // 对角线长度
	cout << "(" << center.getX() << "," << center.getY() << ")" << " " << "长度：" << fixed << setprecision(2) << diagonal << endl; // 输出时diagonal保留两位小数

	armor.Armor_Point(armor.point, armor.width, armor.height); //四个顶点坐标

	system("pause");

	return 0;
}