#pragma once
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;
using namespace cv;

// 全局配置参数统一管理
namespace Config
{
	// 预处理参数
	extern int BINARY_THRESHHOLD; // 二值化阈值
	extern int morph_kernel_size; // 膨胀核大小
	extern int morph_iterations; // 膨胀迭代次数

	// 中心R识别参数
	extern double r_center_min_area; // 最小面积
	extern double r_center_max_area; // 最大面积
	extern double r_center_min_ratio; // 最小长宽比
	extern double r_center_max_ratio; // 最大长宽比

	// 扇叶检测参数
	extern double fan_blade_min_area; // 扇叶最小面积
	extern double fan_blade_max_area; // 扇叶最大面积
	extern double fan_blade_min_ratio; // 扇叶最小长宽比
	extern double fan_blade_max_ratio; // 扇叶最大长宽比

	//extern float raduis; // 能量机关半径（扇叶中心到中心R的距离）
};

// 简单计时器类（用于计算各个模块的耗时和FPS）
class SimpleTimer
{
private:
	chrono::high_resolution_clock::time_point start_time; // 记录开始时间
public:
	void start() // 开始计时
	{
		start_time = chrono::high_resolution_clock::now();
	}
	double getElapsedMs() // 计算耗时
	{
		auto end_time = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
		return duration.count() / 1000.0f; // 转换为毫秒
	}
	void printElapsed(const string& name) // 打印耗时
	{
		cout << name << ": " << getElapsedMs() << "ms" << std::endl;
	}
};