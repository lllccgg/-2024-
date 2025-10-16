#include <iostream>

using namespace std;

class Point//����
{
public:
    //ʹ�ó�ʼ�����ʼ������
    Point(int a = 0, int b = 0) :x(a), y(b) {}
    int getX() const;//�õ�x����
    int getY() const;//�õ�y����

    //����<<ʵ�ֵ����������
    friend ostream& operator<<(ostream& output, const Point& p);

protected:
    int x;//x����
    int y;//y����
};

//�õ�x��ֵ
int Point::getX() const
{
    return x;
}

//�õ�y��ֵ
int Point::getY() const
{
    return y;
}

//����<<ʵ�ֵ����������
ostream& operator<<(ostream& output, const Point& p)
{
    output << "(" << p.x << "," << p.y << ")" << endl;

    return output;
}
