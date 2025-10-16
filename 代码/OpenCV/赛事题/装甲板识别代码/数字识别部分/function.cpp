#include "function.hpp"

// ��ֵ����ֵ
void BinaryThreshHoldCallback(int value, void* userdata)
{
    Config::BINARY_THRESHOLD = static_cast<float>(value);
}
// ��С���Ƴ����
void MinLightRatioCallback(int value, void* userdata)
{
    Config::MIN_LIGHT_RATIO = static_cast<double>(value / 100.0f);
}
// ������Ƴ����
void MaxLightRatioCallback(int value, void* userdata)
{
    Config::MAX_LIGHT_RATIO = static_cast<double>(value / 100.0f);
}
// ������ƽǶ�ƫ��
void MaxLightAngleDiffCallback(int value, void* userdata)
{
    Config::MAX_ANGLE_DIFF = static_cast<double>(value / 10.0f);
}
// ����������
void MaxLightAreaCallback(int value, void* userdata)
{
    Config::MAX_LIGHT_AREA = static_cast<float>(value); 
}
// ��С�������
void MinLightAreaCallback(int value, void* userdata)
{
    Config::MIN_LIGHT_AREA = static_cast<float>(value);
}
// װ�װ����ǶȲ�
void MaxArmorAngleDiffCallback(int value, void* userdata)
{
    Config::MAX_ARMOR_ANGLE_DIFF = static_cast<float>(value / 10.0f);
}
// װ�װ���󳤶ȱȲ��죨�������Ƶĳ��ȱȣ�
void MaxLengthRatioCallback(int value, void* userdata)
{
    Config::MAX_LENGTH_RATIO = static_cast<double>(value / 10.0f);
}
// װ�װ�����λ�ǣ��ȣ�
void MaxDisplacementAngleCallback(int value, void* userdata)
{
    Config::MAX_DISPLACEMENT_ANGLE = static_cast<double>(value / 10.0f);
}
// ��Сװ�װ��߱�
void MinArmorAspectRatioCallback(int value, void* userdata)
{
    Config::MIN_ARMOR_ASPECT_RATIO = static_cast<double>(value / 10.0f);
}
// ���װ�װ��߱�
void MaxArmorAspectRatioCallback(int value, void* userdata)
{
    Config::MAX_ARMOR_ASPECT_RATIO = static_cast<double>(value / 10.0f);
}
// Ԥ�����ֵ����ֵ
void binary1callback(int value, void* userdata)
{
    Config::BINARY_THRESHOLD = value;
}
// ����ģ�͵�ͼ���ֵ����ֵ
void binary2callback(int value, void* userdata)
{
    Config::bianry = value;
}
// ������˴�С
void kernel1callback(int value, void* userdata)
{
    Config::kernel1_size = value+1;
}
// ��ʴ�˴�С
void kernel2callback(int value, void* userdata)
{
    Config::kernel2_size = value+1;
}
// ROI����x����
void roixcallback(int value, void* userdata)
{
    Config::roi_x = value;
}
// ROI����y����
void roiycallback(int value, void* userdata)
{
    Config::roi_y = value;
}
// x����ƫ��
void x_offsetcallback(int value, void* userdata)
{
    Config::x_offset = value - 50;
}
// y����ƫ��
void y_offsetcallback(int value, void* userdata)
{
    Config::y_offset = value - 50;
}
// �����ع�٤��ֵ
void gamma_callback(int value, void* userdata)
{
    Config::gamma_val = value;
}
// �����ع����Ƚض���ֵ
void trunc_callback(int value, void* userdata)
{
    Config::trunc_val = value;
}
// ��ʴ������������
void interations_callback(int value, void* userdata)
{
    Config::interations = value + 1;
}
// �������2��С
void kernel3callback(int value, void* userdata)
{
    Config::kernel3_size = value + 1;
}
// ͸��ƫ��
void warp_diff_callback(int value, void* userdata)
{
    Config::warp_diff = value / 200.0f;
}
// ��ֱ�ߺ˵�һ������ 
void line1callback(int value, void* userdata)
{
    Config::line1 = value + 1;
}
// ��ֱ�ߺ˵ڶ�������
void line2callback(int value, void* userdata)
{
    Config::line2 = value + 1;
}
// ����˵�һ������
void clean1callback(int value, void* userdata)
{
    Config::clean1 = value + 1;
}
// ����˵ڶ�������
void clean2callback(int value, void* userdata)
{
    Config::clean2 = value + 1;
}
// HSV����H
void hsv_h_low_callback(int value, void* userdata)
{
    Config::h_low = value;
}
// HSV����S
void hsv_s_low_callback(int value, void* userdata)
{
    Config::s_low = value;
}
// HSV����V
void hsv_v_low_callback(int value, void* userdata)
{
    Config::v_low = value;
}
// HSV����H
void hsv_h_high_callback(int value, void* userdata)
{
    Config::h_high = value;
}
// HSV����S
void hsv_s_high_callback(int value, void* userdata)
{
    Config::s_high = value;
}
// HSV����V
void hsv_v_high_callback(int value, void* userdata)
{
    Config::v_high = value;
}

void Trackbar(int n, string windowname)
{
    //createTrackbar("��ֵ����ֵ", windowname, NULL, 255, BinaryThreshHoldCallback);
    createTrackbar("L��С�����", windowname, NULL, 1000, MinLightRatioCallback);
    createTrackbar("L��󳤿��", windowname, NULL, 10000, MaxLightRatioCallback);
    createTrackbar("L��С���", windowname, NULL, 1000, MinLightAreaCallback);
    createTrackbar("L������", windowname, NULL, 100000, MaxLightAreaCallback);

    createTrackbar("��С��߱�", windowname, NULL, 100, MinArmorAspectRatioCallback);
    createTrackbar("����߱�", windowname, NULL, 100, MaxArmorAspectRatioCallback);
    createTrackbar("��󳤶ȱȲ�", windowname, NULL, 100, MaxLengthRatioCallback);
    createTrackbar("����λ��", windowname, NULL, 360, MaxDisplacementAngleCallback);
    createTrackbar("���ǶȲ�", windowname, NULL, 360, MaxArmorAngleDiffCallback);
 //   createTrackbar("��ֵ��1", windowname, NULL, 255, binary1callback);
 //   createTrackbar("��ֵ��2", windowname, NULL, 255, binary2callback);
	//createTrackbar("�������1", windowname, NULL, 20, kernel1callback);
	//createTrackbar("��ʴ��", windowname, NULL, 20, kernel2callback);
	//createTrackbar("�������2", windowname, NULL, 20, kernel3callback);
	//createTrackbar("roi_x", windowname, NULL, 100, roixcallback);
	//createTrackbar("roi_y", windowname, NULL, 100, roiycallback);
	//createTrackbar("x_offset", windowname, NULL, 200, x_offsetcallback);
	//createTrackbar("y_offset", windowname, NULL, 200, y_offsetcallback);
	//createTrackbar("gamma", windowname, NULL, 50, gamma_callback);
	//createTrackbar("trunc", windowname, NULL, 255, trunc_callback);
	//createTrackbar("��ʴ����", windowname, NULL, 10, interations_callback);
	//createTrackbar("͸��ƫ��", windowname, NULL, 100, warp_diff_callback);
	//createTrackbar("line1", windowname, NULL, 20, line1callback);
	//createTrackbar("line2", windowname, NULL, 20, line2callback);
	//createTrackbar("clean1", windowname, NULL, 20, clean1callback);
	//createTrackbar("clean2", windowname, NULL, 20, clean2callback);
	createTrackbar("H_low", windowname, NULL, 180, hsv_h_low_callback);
	createTrackbar("S_low", windowname, NULL, 255, hsv_s_low_callback);
	createTrackbar("V_low", windowname, NULL, 255, hsv_v_low_callback);
	createTrackbar("H_high", windowname, NULL, 180, hsv_h_high_callback);
	createTrackbar("S_high", windowname, NULL, 255, hsv_s_high_callback);
	createTrackbar("V_high", windowname, NULL, 255, hsv_v_high_callback);
}
