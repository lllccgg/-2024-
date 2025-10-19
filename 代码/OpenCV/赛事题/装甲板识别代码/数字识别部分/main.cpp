#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "config.hpp"
#include "function.hpp"
#include "LightBar.hpp"
#include "ArmorDetected.hpp"

using namespace cv;
using namespace std;

// 全局变量
namespace Config {
    int BINARY_THRESHOLD = 130; // 预处理二值化阈值

    int h_low = 15; // HSV下限H
    int s_low = 0; // HSV下限S
    int v_low = 15; // HSV下限V
    int h_high = 130; // HSV上限H
    int s_high = 175; // HSV上限S
    int v_high = 255; // HSV上限V

    Scalar lower_blue(Config::h_low, Config::s_low, Config::v_low);   // H, S, V 下限
    Scalar upper_blue(Config::h_high, Config::s_high, Config::v_high); // H, S, V 上限

	int kernel1_size = 1; // 闭运算核大小
	int kernel2_size = 3; // 腐蚀核大小
	int kernel3_size = 3; // 闭运算核2大小
	int interations = 1; // 腐蚀操作迭代次数

    int line1 = 3; // 垂直线核第一个参数
    int line2 = 1; // 垂直线核第二个参数
    int clean1 = 1; // 清理核第一个参数
	int clean2 = 2; // 清理核第二个参数

    Mat kernel_line = getStructuringElement(MORPH_RECT, Size(Config::line1, Config::line2)); // 垂直线核

    double MIN_LIGHT_RATIO = 0.01; // 最小灯条长宽比
    double MAX_LIGHT_RATIO = 20.0; // 最大灯条长宽比
    double MAX_ANGLE_DIFF = 20.0f; // 条灯最大角度偏差 
    float MIN_LIGHT_AREA = 0.0f; // 条灯最小面积
    float MAX_LIGHT_AREA = 20000.0f; // 条灯最大面积
    float MAX_ARMOR_ANGLE_DIFF = 20.0f; // 装甲板最大角度差
    float MAX_LENGTH_RATIO = 0.7f; // 装甲板最大长度比差异（两个条灯的长度比）
    float MAX_DISPLACEMENT_ANGLE = 50.0f; // 装甲板最大错位角（度）
    float MIN_ARMOR_ASPECT_RATIO = 1.5f; // 最小装甲板宽高比
    float MAX_ARMOR_ASPECT_RATIO = 30.0f; // 最大装甲板宽高比

	const char* const onnx_path = "D:\\vsworkspaces\\ArmorNumClassifier_svm_model\\bestv11.onnx"; // onnx模型路径

    int bianry = 1; // 输入模型的图像二值化阈值
	int roi_x = 16; // ROI区域x方向
	int roi_y = 25; // ROI区域y方向
	int x_offset = 16; // x方向偏移
	int y_offset = 55; // y方向偏移

	// 调节曝光
    int gamma_val = 13; // 初始伽马值（x10，例如1.8）
    int trunc_val = 225; // 初始亮度截断阈值

    float warp_diff = 0.5; // 透视偏差（百分比）

    // 稳定化参数初始化
    int STABILIZE_HISTORY_SIZE = 4;        // 历史帧数
    float STABILIZE_WEIGHT = 0.7f;         // 当前帧权重  
    // float MAX_POSITION_JUMP = 200.0f;       // 最大位置跳跃
    // float MAX_SIZE_CHANGE = 0.1f;          // 最大尺寸变化率
}

VideoCapture cap;
   
int main()
{
    cv::setUseOptimized(true);  // 启用OpenCV优化
    cv::setNumThreads(8);       // 设置线程数为CPU核心数

    cap.open("D:\\Opencvpracticedata\\202510172048.mp4"); // 读取视频路径
    

    Mat srcImg;
    if (!cap.read(srcImg))
    {
        cout << "视频有误！" << endl;
        return - 1;
    }

	// 创建窗口和滑动条
    namedWindow("trackbar", WINDOW_GUI_NORMAL);
    Trackbar(1, "trackbar");

    int frameCount = 0; // 获取的帧数
    int framecount = 0; // 一直记录帧数（只是用于在起始帧时进行selectROI）
    SimpleTimer fpsTimer; // 用于计算FPS的计时器
    fpsTimer.start();
    double fps = 0.0f; // 存储计算得到的FPS
    string str = ""; // 用于显示的字符串

	ArmorDetected detector; // 实例化装甲板检测对象

	Rect l_light_roi; // 左侧灯条ROI区域
	Rect r_light_roi; // 右侧灯条ROI区域

    while (true)
    {   
        if (!cap.read(srcImg))
        {
            break;
        }
		Mat img = srcImg.clone();
        resize(img, img, Size(360, 640));
        // 降曝光模拟（伽马 + 裁亮）
        srcImg.convertTo(img, CV_32F, 1.0 / 255.0);
        cv::pow(img, Config::gamma_val, img); // gamma校正
        img.convertTo(img, CV_8U, 255.0);
        // 限制高亮
        cv::threshold(img, img, Config::trunc_val, Config::trunc_val, cv::THRESH_TRUNC);
        

        if (framecount == 0) // 第0帧进行初始化
        {
			//resize(img, img, Size(540, 960));
            // 手动选择左右灯条区域
            l_light_roi = selectROI("Select Left Light", img);
            r_light_roi = selectROI("Select Right Light", img);

            // 将ROI转换为BBox并初始化last_lightBars
            if (l_light_roi.width > 0 && r_light_roi.width > 0) 
            {
                detector.initializeLastLightBars(l_light_roi, r_light_roi);
            }
        }

		Mat debugImg = img.clone(); // 用于显示识别的效果
        
        // 进行装甲板识别
        bool detected = detector.detectArmor(img);

        framecount++;
        frameCount++;
        double elapsed = fpsTimer.getElapsedMs();
        
        // 每秒计算一次FPS
        if (elapsed >= 1000.0f)
        {
            fps = frameCount / (elapsed / 1000.0f);
            frameCount = 0;
            fpsTimer.start();
            str = "FPS:" + to_string(fps); // 用于实时显示FPS
        }
        putText(debugImg, str, Point2f(100, 100), FONT_HERSHEY_SIMPLEX, 0.5, Config::COLOR_GREEN);

		// 如果检测到装甲板
        if (detected)
        {
            // 显示
            detector.showLightBars(debugImg);
            detector.showArmors(debugImg);
        }

		namedWindow("result", WINDOW_NORMAL);
        imshow("result", debugImg);

        if (char(waitKey(1)) == 27)
        {
            break;
        }
    }

    waitKey(0);
    return 0;
}
