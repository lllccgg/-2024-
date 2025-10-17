#pragma once
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;
using namespace cv;

// 全局配置参数统一管理
namespace Config {
    // 图像处理参数
    const int IMAGE_WIDTH = 640; // 图像宽度
    const int IMAGE_HEIGHT = 480; // 图像高度
    extern int BINARY_THRESHOLD; // 二值化阈值

	extern int h_low; // HSV下限H
	extern int s_low; // HSV下限S
	extern int v_low; // HSV下限V
	extern int h_high; // HSV上限H
	extern int s_high; // HSV上限S
	extern int v_high; // HSV上限V

    extern Scalar lower_blue;   // H, S, V 下限
    extern Scalar upper_blue; // H, S, V 上限

	extern int kernel1_size; // 闭运算核大小
	extern int kernel2_size; // 开运算核大小
    extern int kernel3_size; // 闭运算核2大小

    extern int line1; // 垂直线核第一个参数
	extern int line2; // 垂直线核第二个参数
	extern int clean1; // 清理核大小
	extern int clean2; // 清理核大小

    extern Mat kernel_line; // 垂直线核

    extern int interations; // 腐蚀操作迭代次数

    // 装甲板检测参数
    extern double MIN_LIGHT_RATIO; // 最小灯条长宽比
    extern double MAX_LIGHT_RATIO; // 最大灯条长宽比
    extern double MAX_ANGLE_DIFF; // 条灯最大角度偏差 

    // 装甲板匹配参数
    extern float MAX_ARMOR_ANGLE_DIFF; // 最大角度差（度）
    extern float MAX_LENGTH_RATIO; // 最大长度比差异（两个条灯的长度比）
    extern float MAX_DISPLACEMENT_ANGLE; // 最大错位角（度）
    extern float MIN_ARMOR_ASPECT_RATIO; // 最小装甲板宽高比
    extern float MAX_ARMOR_ASPECT_RATIO; // 最大装甲板宽高比
    extern float MIN_LIGHT_AREA; // 最小面积
    extern float MAX_LIGHT_AREA; // 最大面积


    // 调试显示颜色
    const Scalar COLOR_GREEN = Scalar(0, 255, 0); // 绿色
    const Scalar COLOR_RED = Scalar(0, 0, 255); // 红色
    const Scalar COLOR_BLUE = Scalar(255, 0, 0); // 蓝色
    const Scalar COLOR_YELLOW = Scalar(0, 255, 255); // 黄色
    const Scalar COLOR_CYAN = Scalar(255, 255, 0); // 青色（新增）
    const Scalar COLOR_MAGENTA = Scalar(255, 0, 255); // 洋红色（新增）
    const Scalar COLOR_WHITE = Scalar(255, 255, 255); // 白色

    // 相机的3×3内参矩阵
    const Mat cameraMatrix = (Mat_<double>(3, 3) <<
        619.2342625843423, 0.                , 332.4107997008066, // fx, 0, cx
        0.               , 619.4356974250945 , 237.9417298019534, // 0, fy, cy  
        0.               , 0.                , 1.                 // 0, 0, 1
        );

    // 5个畸变系数：k1, k2, p1, p2, k3
    const Mat distCoeffs = (Mat_<double>(5, 1) <<
        -0.1857965552869972, 0.7150658975139499, 0.003183596780102963, 0.0009642865213194783, -0.8848275920367117
        );

    // 装甲板实际尺寸(毫米)
    const double ARMOR_WIDTH = 145.0; // 装甲板宽度
    const double LIGHT_HEIGHT = 65.0; // 条灯高度

	extern const char* const onnx_path; // onnx模型路径

    extern int bianry; // 输入模型的图像二值化阈值

    extern int roi_x; // 改变输入模型的图像x方向大小
	extern int roi_y; // 改变输入模型的图像y方向大小

	extern int x_offset; // x方向偏移
	extern int y_offset; // y方向偏移

    // 调节曝光
    extern int gamma_val; // 初始伽马值（x10，例如1.8）
    extern int trunc_val; // 初始亮度截断阈值

    extern float warp_diff; // 透视偏差

    // 添加稳定化参数
    extern int STABILIZE_HISTORY_SIZE;     // 历史帧数
    extern float STABILIZE_WEIGHT;         // 当前帧权重
    extern float MAX_POSITION_JUMP;        // 最大位置跳跃
    extern float MAX_SIZE_CHANGE;          // 最大尺寸变化率
}

// 简单计时器类
class SimpleTimer
{
private:
    chrono::high_resolution_clock::time_point start_time;
public:
    void start()
    {
        start_time = chrono::high_resolution_clock::now();
    }
    double getElapsedMs()
    {
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0f; // 转换为毫秒
    }
    void printElapsed(const string& name)
    {
        cout << name << ": " << getElapsedMs() << "ms" << std::endl;
    }
};
