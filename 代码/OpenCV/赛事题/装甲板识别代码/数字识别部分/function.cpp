#include "function.hpp"

// 二值化阈值
void BinaryThreshHoldCallback(int value, void* userdata)
{
    Config::BINARY_THRESHOLD = static_cast<float>(value);
}
// 最小条灯长宽比
void MinLightRatioCallback(int value, void* userdata)
{
    Config::MIN_LIGHT_RATIO = static_cast<double>(value / 100.0f);
}
// 最大条灯长宽比
void MaxLightRatioCallback(int value, void* userdata)
{
    Config::MAX_LIGHT_RATIO = static_cast<double>(value / 100.0f);
}
// 最大条灯角度偏差
void MaxLightAngleDiffCallback(int value, void* userdata)
{
    Config::MAX_ANGLE_DIFF = static_cast<double>(value / 10.0f);
}
// 最大条灯面积
void MaxLightAreaCallback(int value, void* userdata)
{
    Config::MAX_LIGHT_AREA = static_cast<float>(value); 
}
// 最小条灯面积
void MinLightAreaCallback(int value, void* userdata)
{
    Config::MIN_LIGHT_AREA = static_cast<float>(value);
}
// 装甲板最大角度差
void MaxArmorAngleDiffCallback(int value, void* userdata)
{
    Config::MAX_ARMOR_ANGLE_DIFF = static_cast<float>(value / 10.0f);
}
// 装甲板最大长度比差异（两个条灯的长度比）
void MaxLengthRatioCallback(int value, void* userdata)
{
    Config::MAX_LENGTH_RATIO = static_cast<double>(value / 10.0f);
}
// 装甲板最大错位角（度）
void MaxDisplacementAngleCallback(int value, void* userdata)
{
    Config::MAX_DISPLACEMENT_ANGLE = static_cast<double>(value / 10.0f);
}
// 最小装甲板宽高比
void MinArmorAspectRatioCallback(int value, void* userdata)
{
    Config::MIN_ARMOR_ASPECT_RATIO = static_cast<double>(value / 10.0f);
}
// 最大装甲板宽高比
void MaxArmorAspectRatioCallback(int value, void* userdata)
{
    Config::MAX_ARMOR_ASPECT_RATIO = static_cast<double>(value / 10.0f);
}
// 预处理二值化阈值
void binary1callback(int value, void* userdata)
{
    Config::BINARY_THRESHOLD = value;
}
// 输入模型的图像二值化阈值
void binary2callback(int value, void* userdata)
{
    Config::bianry = value;
}
// 闭运算核大小
void kernel1callback(int value, void* userdata)
{
    Config::kernel1_size = value+1;
}
// 腐蚀核大小
void kernel2callback(int value, void* userdata)
{
    Config::kernel2_size = value+1;
}
// ROI区域x方向
void roixcallback(int value, void* userdata)
{
    Config::roi_x = value;
}
// ROI区域y方向
void roiycallback(int value, void* userdata)
{
    Config::roi_y = value;
}
// x方向偏移
void x_offsetcallback(int value, void* userdata)
{
    Config::x_offset = value - 50;
}
// y方向偏移
void y_offsetcallback(int value, void* userdata)
{
    Config::y_offset = value - 50;
}
// 调节曝光伽马值
void gamma_callback(int value, void* userdata)
{
    Config::gamma_val = value;
}
// 调节曝光亮度截断阈值
void trunc_callback(int value, void* userdata)
{
    Config::trunc_val = value;
}
// 腐蚀操作迭代次数
void interations_callback(int value, void* userdata)
{
    Config::interations = value + 1;
}
// 闭运算核2大小
void kernel3callback(int value, void* userdata)
{
    Config::kernel3_size = value + 1;
}
// 透视偏差
void warp_diff_callback(int value, void* userdata)
{
    Config::warp_diff = value / 200.0f;
}
// 垂直线核第一个参数 
void line1callback(int value, void* userdata)
{
    Config::line1 = value + 1;
}
// 垂直线核第二个参数
void line2callback(int value, void* userdata)
{
    Config::line2 = value + 1;
}
// 清理核第一个参数
void clean1callback(int value, void* userdata)
{
    Config::clean1 = value + 1;
}
// 清理核第二个参数
void clean2callback(int value, void* userdata)
{
    Config::clean2 = value + 1;
}
// HSV下限H
void hsv_h_low_callback(int value, void* userdata)
{
    Config::h_low = value;
}
// HSV下限S
void hsv_s_low_callback(int value, void* userdata)
{
    Config::s_low = value;
}
// HSV下限V
void hsv_v_low_callback(int value, void* userdata)
{
    Config::v_low = value;
}
// HSV上限H
void hsv_h_high_callback(int value, void* userdata)
{
    Config::h_high = value;
}
// HSV上限S
void hsv_s_high_callback(int value, void* userdata)
{
    Config::s_high = value;
}
// HSV上限V
void hsv_v_high_callback(int value, void* userdata)
{
    Config::v_high = value;
}

void Trackbar(int n, string windowname)
{
    //createTrackbar("二值化阈值", windowname, NULL, 255, BinaryThreshHoldCallback);
    createTrackbar("L最小长宽比", windowname, NULL, 1000, MinLightRatioCallback);
    createTrackbar("L最大长宽比", windowname, NULL, 10000, MaxLightRatioCallback);
    createTrackbar("L最小面积", windowname, NULL, 1000, MinLightAreaCallback);
    createTrackbar("L最大面积", windowname, NULL, 100000, MaxLightAreaCallback);

    createTrackbar("最小宽高比", windowname, NULL, 100, MinArmorAspectRatioCallback);
    createTrackbar("最大宽高比", windowname, NULL, 100, MaxArmorAspectRatioCallback);
    createTrackbar("最大长度比差", windowname, NULL, 100, MaxLengthRatioCallback);
    createTrackbar("最大错位角", windowname, NULL, 360, MaxDisplacementAngleCallback);
    createTrackbar("最大角度差", windowname, NULL, 360, MaxArmorAngleDiffCallback);
 //   createTrackbar("二值化1", windowname, NULL, 255, binary1callback);
 //   createTrackbar("二值化2", windowname, NULL, 255, binary2callback);
	//createTrackbar("闭运算核1", windowname, NULL, 20, kernel1callback);
	//createTrackbar("腐蚀核", windowname, NULL, 20, kernel2callback);
	//createTrackbar("闭运算核2", windowname, NULL, 20, kernel3callback);
	//createTrackbar("roi_x", windowname, NULL, 100, roixcallback);
	//createTrackbar("roi_y", windowname, NULL, 100, roiycallback);
	//createTrackbar("x_offset", windowname, NULL, 200, x_offsetcallback);
	//createTrackbar("y_offset", windowname, NULL, 200, y_offsetcallback);
	//createTrackbar("gamma", windowname, NULL, 50, gamma_callback);
	//createTrackbar("trunc", windowname, NULL, 255, trunc_callback);
	//createTrackbar("腐蚀迭代", windowname, NULL, 10, interations_callback);
	//createTrackbar("透视偏差", windowname, NULL, 100, warp_diff_callback);
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
