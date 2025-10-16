#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "config.hpp"
#include "function.hpp"
#include "LightBar.hpp"
#include "ArmorDetected.hpp"

using namespace cv;
using namespace std;

// ȫ�ֱ���
namespace Config {
    int BINARY_THRESHOLD = 130; // Ԥ�����ֵ����ֵ

    int h_low = 15; // HSV����H
    int s_low = 0; // HSV����S
    int v_low = 50; // HSV����V
    int h_high = 130; // HSV����H
    int s_high = 175; // HSV����S
    int v_high = 255; // HSV����V

    Scalar lower_blue(Config::h_low, Config::s_low, Config::v_low);   // H, S, V ����
    Scalar upper_blue(Config::h_high, Config::s_high, Config::v_high); // H, S, V ����

    int kernel1_size = 1; // ������˴�С
    int kernel2_size = 3; // ��ʴ�˴�С
    int kernel3_size = 3; // �������2��С
    int interations = 1; // ��ʴ������������

    int line1 = 16; // ��ֱ�ߺ˵�һ������
    int line2 = 1; // ��ֱ�ߺ˵ڶ�������
    int clean1 = 1; // ����˵�һ������
    int clean2 = 2; // ����˵ڶ�������

    Mat kernel_line = getStructuringElement(MORPH_RECT, Size(Config::line1, Config::line2)); // ��ֱ�ߺ�

    double MIN_LIGHT_RATIO = 0.01; // ��С���������
    double MAX_LIGHT_RATIO = 15.0; // �����������
    double MAX_ANGLE_DIFF = 20.0f; // �������Ƕ�ƫ�� 
    float MIN_LIGHT_AREA = 500.0f; // ������С���
    float MAX_LIGHT_AREA = 20000.0f; // ����������
    float MAX_ARMOR_ANGLE_DIFF = 20.0f; // װ�װ����ǶȲ�
    float MAX_LENGTH_RATIO = 0.7f; // װ�װ���󳤶ȱȲ��죨�������Ƶĳ��ȱȣ�
    float MAX_DISPLACEMENT_ANGLE = 50.0f; // װ�װ�����λ�ǣ��ȣ�
    float MIN_ARMOR_ASPECT_RATIO = 1.5f; // ��Сװ�װ��߱�
    float MAX_ARMOR_ASPECT_RATIO = 30.0f; // ���װ�װ��߱�

    int bianry = 8; // ����ģ�͵�ͼ���ֵ����ֵ
    int roi_x = 30; // ROI����x����
    int roi_y = 45; // ROI����y����
    int x_offset = 65; // x����ƫ��
    int y_offset = 115; // y����ƫ��

    // �����ع�
    int gamma_val = 8; // ��ʼ٤��ֵ��x10������1.8��
    int trunc_val = 220; // ��ʼ���Ƚض���ֵ

    float warp_diff = 0.5; // ͸��ƫ��ٷֱȣ�

    // �ȶ���������ʼ��
    int STABILIZE_HISTORY_SIZE = 4;        // ��ʷ֡��
    float STABILIZE_WEIGHT = 0.7f;         // ��ǰ֡Ȩ��  
    float MAX_POSITION_JUMP = 200.0f;       // ���λ����Ծ
    float MAX_SIZE_CHANGE = 0.1f;          // ���ߴ�仯��
}

VideoCapture cap;

int main()
{
    cv::setUseOptimized(true);  // ����OpenCV�Ż�
    cv::setNumThreads(8);       // �����߳���ΪCPU������

    cap.open(0); // ������ͷ

    // ���÷ֱ���
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    Mat srcImg;
    if (!cap.read(srcImg))
    {
        cout << "��Ƶ����" << endl;
        return -1;
    }

    // �������ںͻ�����
    namedWindow("trackbar", WINDOW_GUI_NORMAL);
    Trackbar(1, "trackbar");

    int frameCount = 0; // ��ȡ��֡��
    int framecount = 0; // һֱ��¼֡����ֻ����������ʼ֡ʱ����selectROI��
    SimpleTimer fpsTimer; // ���ڼ���FPS�ļ�ʱ��
    fpsTimer.start();
    double fps = 0.0f; // �洢����õ���FPS
    string str = ""; // ������ʾ���ַ���

    ArmorDetected detector; // ʵ����װ�װ������

    Rect l_light_roi; // ������ROI����
    Rect r_light_roi; // �Ҳ����ROI����


    while (true)
    {
        if (!cap.read(srcImg))
        {
            break;
        }
        Mat img = srcImg.clone();
        // ���ع�ģ�⣨٤�� + ������
        srcImg.convertTo(img, CV_32F, 1.0 / 255.0);
        cv::pow(img, Config::gamma_val, img); // gammaУ��
        img.convertTo(img, CV_8U, 255.0);
        // ���Ƹ���
        cv::threshold(img, img, Config::trunc_val, Config::trunc_val, cv::THRESH_TRUNC);


        if (framecount == 0) // ��0֡���г�ʼ��
        {
            //resize(img, img, Size(540, 960));
            // �ֶ�ѡ�����ҵ�������
            l_light_roi = selectROI("Select Left Light", img);
            r_light_roi = selectROI("Select Right Light", img);

            // ��ROIת��ΪBBox����ʼ��last_lightBars
            if (l_light_roi.width > 0 && r_light_roi.width > 0)
            {
                detector.initializeLastLightBars(l_light_roi, r_light_roi);
            }
        }

        Mat debugImg = img.clone(); // ������ʾʶ���Ч��

        // ����װ�װ�ʶ��
        bool detected = detector.detectArmor(img);

        framecount++;
        frameCount++;
        double elapsed = fpsTimer.getElapsedMs();

        // ÿ�����һ��FPS
        if (elapsed >= 1000.0f)
        {
            fps = frameCount / (elapsed / 1000.0f);
            frameCount = 0;
            fpsTimer.start();
            str = "FPS:" + to_string(fps); // ����ʵʱ��ʾFPS
        }
        putText(debugImg, str, Point2f(100, 100), FONT_HERSHEY_SIMPLEX, 0.5, Config::COLOR_GREEN);

        // �����⵽װ�װ�
        if (detected)
        {
            // ��ʾ
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
