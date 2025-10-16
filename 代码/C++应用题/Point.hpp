#include <iostream>

using namespace std;

class Point//点类
{
public:
    //使用初始化表初始化点类
    Point(int a = 0, int b = 0) :x(a), y(b) {}
    int getX() const;//得到x坐标
    int getY() const;//得到y坐标

    //重载<<实现点的坐标的输出
    friend ostream& operator<<(ostream& output, const Point& p);

protected:
    int x;//x坐标
    int y;//y坐标
};

//得到x的值
int Point::getX() const
{
    return x;
}

//得到y的值
int Point::getY() const
{
    return y;
}

//重载<<实现点的坐标的输出
ostream& operator<<(ostream& output, const Point& p)
{
    output << "(" << p.x << "," << p.y << ")" << endl;

    return output;
}
