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

class Armor : public Rect //�̳�
{
public:
	Armor(const Rect& rect) : Rect(rect) {} // ��ʼ���б�

	Point Central_Point(const Point point, int width, int height); // �������������
	double Diagonal(int width, int height); // ������Խ��߳���
	void Armor_Point(const Point point, int width, int height); // ����ĸ���������
	void Armor_Color(int color); // ����궨����ɫ
};

// �������������
Point Armor::Central_Point(const Point point, int width, int height)
{
	Point center = Point(point.getX() + width / 2, point.getY() + height / 2);
	return center;
}

// ������Խ��߳���
double Armor::Diagonal(int width, int height)
{
	double diagonal = sqrt(width * width + height * height);
	return diagonal; 
}

// ����ĸ���������
void Armor::Armor_Point(const Point point, int width, int height)
{
	Point left_top = point; // ����
	Point right_top = Point(point.getX() + width, point.getY()); // ����
	Point right_bottom = Point(point.getX() + width, point.getY() + height); // ����
	Point left_bottom = Point(point.getX(), point.getY() + height); // ����
	
	cout << "(" << left_top.getX() << "," << left_top.getY() << ")" << " "
		<< "(" << right_top.getX() << "," << right_top.getY() << ")" << " "
		<< "(" << right_bottom.getX() << "," << right_bottom.getY() << ")" << " "
		<< "(" << left_bottom.getX() << "," << left_bottom.getY() << ")" << endl;
}

// ����궨����ɫ
void Armor::Armor_Color(int color)
{
	if (color) // 1Ϊ��ɫ
	{
		cout << "��ɫ����" << endl;
	}
	else // 0Ϊ��ɫ
	{
		cout << "��ɫ����" << endl;
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

	cout << "ID��" << rect.numID << " "; 
	armor.Armor_Color(armor.color);

	Point center = armor.Central_Point(armor.point, armor.width, armor.height); // ��������
	double diagonal = armor.Diagonal(armor.width, armor.height); // �Խ��߳���
	cout << "(" << center.getX() << "," << center.getY() << ")" << " " << "���ȣ�" << fixed << setprecision(2) << diagonal << endl; // ���ʱdiagonal������λС��

	armor.Armor_Point(armor.point, armor.width, armor.height); //�ĸ���������

	system("pause");

	return 0;
}