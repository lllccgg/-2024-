#pragma once
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;
using namespace cv;

// ȫ�����ò���ͳһ����
namespace Config {
    // ͼ�������
    const int IMAGE_WIDTH = 640; // ͼ����
    const int IMAGE_HEIGHT = 480; // ͼ��߶�
    extern int BINARY_THRESHOLD; // ��ֵ����ֵ

    extern int h_low; // HSV����H
    extern int s_low; // HSV����S
    extern int v_low; // HSV����V
    extern int h_high; // HSV����H
    extern int s_high; // HSV����S
    extern int v_high; // HSV����V

    extern Scalar lower_blue;   // H, S, V ����
    extern Scalar upper_blue; // H, S, V ����

    extern int kernel1_size; // ������˴�С
    extern int kernel2_size; // ������˴�С
    extern int kernel3_size; // �������2��С

    extern int line1; // ��ֱ�ߺ˵�һ������
    extern int line2; // ��ֱ�ߺ˵ڶ�������
    extern int clean1; // ����˴�С
    extern int clean2; // ����˴�С

    extern Mat kernel_line; // ��ֱ�ߺ�

    extern int interations; // ��ʴ������������

    // װ�װ������
    extern double MIN_LIGHT_RATIO; // ��С���������
    extern double MAX_LIGHT_RATIO; // �����������
    extern double MAX_ANGLE_DIFF; // �������Ƕ�ƫ�� 

    // װ�װ�ƥ�����
    extern float MAX_ARMOR_ANGLE_DIFF; // ���ǶȲ�ȣ�
    extern float MAX_LENGTH_RATIO; // ��󳤶ȱȲ��죨�������Ƶĳ��ȱȣ�
    extern float MAX_DISPLACEMENT_ANGLE; // ����λ�ǣ��ȣ�
    extern float MIN_ARMOR_ASPECT_RATIO; // ��Сװ�װ��߱�
    extern float MAX_ARMOR_ASPECT_RATIO; // ���װ�װ��߱�
    extern float MIN_LIGHT_AREA; // ��С���
    extern float MAX_LIGHT_AREA; // ������


    // ������ʾ��ɫ
    const Scalar COLOR_GREEN = Scalar(0, 255, 0); // ��ɫ
    const Scalar COLOR_RED = Scalar(0, 0, 255); // ��ɫ
    const Scalar COLOR_BLUE = Scalar(255, 0, 0); // ��ɫ
    const Scalar COLOR_YELLOW = Scalar(0, 255, 255); // ��ɫ
    const Scalar COLOR_CYAN = Scalar(255, 255, 0); // ��ɫ��������
    const Scalar COLOR_MAGENTA = Scalar(255, 0, 255); // ���ɫ��������
    const Scalar COLOR_WHITE = Scalar(255, 255, 255); // ��ɫ

    // �����3��3�ڲξ���
    const Mat cameraMatrix = (Mat_<double>(3, 3) <<
        619.2342625843423, 0., 332.4107997008066, // fx, 0, cx
        0., 619.4356974250945, 237.9417298019534, // 0, fy, cy  
        0., 0., 1.                 // 0, 0, 1
        );

    // 5������ϵ����k1, k2, p1, p2, k3
    const Mat distCoeffs = (Mat_<double>(5, 1) <<
        -0.1857965552869972, 0.7150658975139499, 0.003183596780102963, 0.0009642865213194783, -0.8848275920367117
        );

    // װ�װ�ʵ�ʳߴ�(����)
    const double ARMOR_WIDTH = 145.0; // װ�װ���
    const double LIGHT_HEIGHT = 65.0; // ���Ƹ߶�

    extern const char* const svm_path; // svmģ��·��
    extern const char* const onnx_path; // onnxģ��·��

    extern int bianry; // ����ģ�͵�ͼ���ֵ����ֵ

    extern int roi_x; // �ı�����ģ�͵�ͼ��x�����С
    extern int roi_y; // �ı�����ģ�͵�ͼ��y�����С

    extern int x_offset; // x����ƫ��
    extern int y_offset; // y����ƫ��

    // �����ع�
    extern int gamma_val; // ��ʼ٤��ֵ��x10������1.8��
    extern int trunc_val; // ��ʼ���Ƚض���ֵ

    extern float warp_diff; // ͸��ƫ��

    // ����ȶ�������
    extern int STABILIZE_HISTORY_SIZE;     // ��ʷ֡��
    extern float STABILIZE_WEIGHT;         // ��ǰ֡Ȩ��
    extern float MAX_POSITION_JUMP;        // ���λ����Ծ
    extern float MAX_SIZE_CHANGE;          // ���ߴ�仯��
}

// �򵥼�ʱ����
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
        return duration.count() / 1000.0f; // ת��Ϊ����
    }
    void printElapsed(const string& name)
    {
        cout << name << ": " << getElapsedMs() << "ms" << std::endl;
    }
};