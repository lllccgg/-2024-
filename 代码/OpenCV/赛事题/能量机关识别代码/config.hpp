#pragma once
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;
using namespace cv;

// ȫ�����ò���ͳһ����
namespace Config
{
	// Ԥ�������
	extern int BINARY_THRESHHOLD; // ��ֵ����ֵ
	extern int morph_kernel_size; // ���ͺ˴�С
	extern int morph_iterations; // ���͵�������

	// ����Rʶ�����
	extern double r_center_min_area; // ��С���
	extern double r_center_max_area; // ������
	extern double r_center_min_ratio; // ��С�����
	extern double r_center_max_ratio; // ��󳤿��

	// ��Ҷ������
	extern double fan_blade_min_area; // ��Ҷ��С���
	extern double fan_blade_max_area; // ��Ҷ������
	extern double fan_blade_min_ratio; // ��Ҷ��С�����
	extern double fan_blade_max_ratio; // ��Ҷ��󳤿��

	//extern float raduis; // �������ذ뾶����Ҷ���ĵ�����R�ľ��룩
};

// �򵥼�ʱ���ࣨ���ڼ������ģ��ĺ�ʱ��FPS��
class SimpleTimer
{
private:
	chrono::high_resolution_clock::time_point start_time; // ��¼��ʼʱ��
public:
	void start() // ��ʼ��ʱ
	{
		start_time = chrono::high_resolution_clock::now();
	}
	double getElapsedMs() // �����ʱ
	{
		auto end_time = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
		return duration.count() / 1000.0f; // ת��Ϊ����
	}
	void printElapsed(const string& name) // ��ӡ��ʱ
	{
		cout << name << ": " << getElapsedMs() << "ms" << std::endl;
	}
};