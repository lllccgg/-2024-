#include "ArmorNumClassifier.hpp"

// ͸�ӱ任��������
void WarpedImage(const cv::Mat& src, cv::Mat& dst, const cv::Point2f srcPoints[4], const cv::Size& dstSize);
// ���ֳ���ȵ�������亯��
Mat preprocessROI(const Mat& roi_img, int target_size);

ArmorNumClassifier::ArmorNumClassifier()
{
	armorImgSize = Size(32, 32); // ����ͼ��ߴ�
	use_onnx = false; // �Ƿ����ONNXģ��
}

void ArmorNumClassifier::loadONNXModel(const char* model_path)
{
	if (!use_onnx)
	{
		try
		{
			onnx_net = dnn::readNetFromONNX(model_path);
			if (onnx_net.empty())
			{
				cout << "ONNXģ�ͼ���ʧ��!" << endl;
				exit(0);
			}
			// ʹ��CPU
			onnx_net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
			onnx_net.setPreferableTarget(dnn::DNN_TARGET_CPU);
			use_onnx = true;
			cout << "ONNXģ�ͼ��سɹ�!" << endl;
			//waitKey(0); // ���� 
		}
		catch (const cv::Exception& e)
		{
			cerr << "Error loading the ONNX model: " << e.what() << endl;
			use_onnx = false;
			exit(0);
		}
	}
}

bool ArmorNumClassifier::getArmorImg(ArmorBox& armor, const Mat& srcImg)
{
	if (srcImg.empty()) 
	{
		std::cerr << "Error: src image is empty!" << std::endl;
		return false;
	}
	// ����ROI���������
	int x = max(0, (int)armor.vertices[0].x + Config::roi_x);
	int y = max(0, (int)armor.vertices[0].y - Config::roi_y);
	int width = min((int)armor.width - Config::x_offset, srcImg.cols - x);
	int height = min((int)armor.height + Config::y_offset, srcImg.rows - y);

	// ���ROI�����Ƿ���Ч
	if (width <= 0 || height <= 0) 
	{
		cout << "ROI����Ŀ�Ȼ�߶���Ч: width=" << width << ", height=" << height << endl;
		return false;
	}
	// �����С�ߴ�Ҫ��
	const int MIN_ROI_SIZE = 5; // ��СROI�ߴ�
	if (width < MIN_ROI_SIZE || height < MIN_ROI_SIZE) 
	{
		cout << "ROI����̫С���޷�����: width=" << width << ", height=" << height
			<< " (��СҪ��: " << MIN_ROI_SIZE << ")" << endl;
		armor.armorNum = -1;
		return false;
	}

	if (x  < 0 || y  < 0 ||
		width  > srcImg.cols ||
		height  > srcImg.rows) 
	{
		cout << "ROI���򳬳���ͼ��߽�" << endl;
		armor.armorNum = -1; // ���Ϊ��Чװ�װ�
		return false;
	}

	if (x + width > srcImg.cols || y + height > srcImg.rows) 
	{
		cout << "ROI���򳬳�ͼ��߽�: x=" << x << ", y=" << y
			<< ", width=" << width << ", height=" << height
			<< ", img_cols=" << srcImg.cols << ", img_rows=" << srcImg.rows << endl;
		armor.armorNum = -1;
		return false;
	}

	Mat roi;
	roi = srcImg(Rect(x, y, width, height)); // ��ȡROI����

	// ���ROI��Ч�Լ��
	if (roi.empty() || roi.cols <= 0 || roi.rows <= 0) 
	{
		cout << "��ȡ��ROI������Ч��Ϊ��" << endl;
		armor.armorNum = -1;
		return false;
	}

	for (int i = 0; i < 4; i++)
	{
		srcPoints[i] = armor.vertices[i] ; // װ�װ嶥������
	}

	warpPerspective_dst = preprocessROI(roi, 32); // �������

	if (!warpPerspective_dst.empty() &&
		warpPerspective_dst.cols > 0 &&
		warpPerspective_dst.rows > 0) 
	{
		namedWindow("Warped Image", WINDOW_GUI_NORMAL);
		imshow("Warped Image", warpPerspective_dst);
	}
	else 
	{
		cout << "���棺������ͼ����Ч��������ʾ" << endl;
	}

	namedWindow("Warped Image", WINDOW_GUI_NORMAL);
	imshow("Warped Image", warpPerspective_dst);
	//imwrite("D:\\Opencvpracticedata\\warp0.png", warpPerspective_dst);
	return true;
}

void ArmorNumClassifier::getArmorNumByONNX(ArmorBox& armor)
{
	// ���ģ���Ƿ���سɹ�
	if (!use_onnx) {
		cout << "ONNXģ��δ����!" << endl;
		return;
	}

	// 1. ����ȡ��ROI�������Ԥ����
	Mat processedImg;
	cvtColor(warpPerspective_dst, processedImg, COLOR_BGR2GRAY);

	if (processedImg.empty() || processedImg.cols <= 0 || processedImg.rows <= 0) {
		std::cerr << "Error: processedImg is invalid for resize!" << std::endl;
		return;
	}

	if (armorImgSize.width <= 0 || armorImgSize.height <= 0) {
		std::cerr << "Error: armorImgSize is invalid!" << std::endl;
		return;
	}

	// 2.�淶ͼ���С
	Mat resized_img;
	if (processedImg.size() != armorImgSize)
	{
		resize(processedImg, resized_img, armorImgSize);
	}
	else
	{
		resized_img = processedImg;
	}
	
	// 3.��ͼ����ж�ֵ��
	Mat binary;
	threshold(resized_img, binary, Config::bianry, 255, THRESH_BINARY_INV);

	// 4.��̬ѧ����
	//Mat kernel = getStructuringElement(MORPH_RECT, Size(1, 2));
	//morphologyEx(binary, binary, MORPH_OPEN, kernel);


	namedWindow("ONNX Input", WINDOW_GUI_NORMAL);
	imshow("ONNX Input", binary);
	//waitKey(0);
	
	double maxval, minval;
	Point maxloc, minloc;
	minMaxLoc(binary, &minval, &maxval, &minloc, &maxloc);
	//cout << "��һ��ǰ��" << " minval=" << minval << ", maxval=" << maxval << endl;

	// 7.��һ��
	binary.convertTo(binary, CV_32F, 1.0 / 255.0);
	
	minMaxLoc(binary, &minval, &maxval, &minloc, &maxloc);
	//cout << "��һ����" << " minval=" << minval << ", maxval=" << maxval << endl;

	// 8. ����4D blob
	Mat blob = dnn::blobFromImage(binary, 1.0, armorImgSize, Scalar(0), false, false);
	//cout << "N=" << blob.size[0] << ", C=" << blob.size[1] << ", H=" << blob.size[2] << ", W=" << blob.size[3] << endl;

	// 9.������������
	onnx_net.setInput(blob);

	// 10.ǰ�򴫲�����ȡ���
	Mat output = onnx_net.forward();



	// 11. ��ȡԤ����
	Point classIdPoint;
	double confidence;
	minMaxLoc(output, nullptr, &confidence, nullptr, &classIdPoint);
	armor.armorNum = classIdPoint.x; // ���������ΪԤ�������
}

// ͸�ӱ任��������
void WarpedImage(const cv::Mat& src, cv::Mat& dst, const cv::Point2f srcPoints[4], const cv::Size& dstSize)
{
	CV_Assert(!src.empty());
	//std::cout << "ԭͼ��С: " << src.cols << "x" << src.rows << "  ����: " << src.type() << std::endl;

	// ���޸�ԭͼ�����ɸ��㸱��
	cv::Mat src_float;
	src.convertTo(src_float, CV_32FC3, 1.0 / 255.0);

	// ��ӡ����ǵ�
	//std::cout << "srcPoints:\n";
	//for (int i = 0; i < 4; ++i)
	//	std::cout << i << ": " << srcPoints[i] << std::endl;

	// ����͸�ӱ任����
	cv::Point2f dstPoints[4] = {
		{0, 0},
		{(float)dstSize.width - 1, 0},
		{(float)dstSize.width - 1, (float)dstSize.height - 1},
		{0, (float)dstSize.height - 1}
	};

	cv::Mat M = cv::getPerspectiveTransform(srcPoints, dstPoints);

	//std::cout << "�任���� M:\n" << M << std::endl;

	// ִ��͸�ӱ任
	cv::warpPerspective(src_float, dst, M, dstSize, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

	// ��������Χ
	double minVal, maxVal;
	cv::minMaxLoc(dst.reshape(1), &minVal, &maxVal);
	//std::cout << "warp�����Χ: " << minVal << " ~ " << maxVal << std::endl;

	// �˻�255��תΪ8λ��ʾ
	dst.convertTo(dst, CV_8UC3, 255.0);

	cv::imshow("warp result", dst);
	//cv::waitKey(0);
}

// ���ֳ���ȵ�������亯��
Mat preprocessROI(const Mat& roi_img, int target_size = 32)
{
	if (roi_img.empty()) {
		std::cerr << "Error: roi_img is empty!" << std::endl;
		return Mat();
	}
	if (target_size <= 0) {
		std::cerr << "Error: invalid target_size = " << target_size << std::endl;
		return Mat();
	}
	if (roi_img.cols <= 0 || roi_img.rows <= 0) {
		std::cerr << "Error: roi_img dimensions invalid! cols=" << roi_img.cols
			<< ", rows=" << roi_img.rows << std::endl;
		return Mat();
	}

	// 1.�������ű��������ֳ����
	double scale = min((double)target_size / roi_img.cols, (double)target_size / roi_img.rows);
	
	if (scale <= 0 || !isfinite(scale)) {
		std::cerr << "Error: invalid scale = " << scale << std::endl;
		return Mat();
	}

	// 2.����������
	int new_width = int(roi_img.cols * scale);
	int new_height = int(roi_img.rows * scale);

	// ȷ����С�ߴ�Ϊ1����
	new_width = max(1, new_width);
	new_height = max(1, new_height);

	// ��ӵ�����Ϣ
	//std::cout << "ROIԭʼ�ߴ�: " << roi_img.cols << "x" << roi_img.rows
	//	<< ", ���ű���: " << scale
	//	<< ", �³ߴ�: " << new_width << "x" << new_height << std::endl;

	// ��ӳߴ���
	if (new_width <= 0 || new_height <= 0) {
		std::cerr << "Error: calculated new dimensions invalid! new_width="
			<< new_width << ", new_height=" << new_height << std::endl;
		return Mat();
	}

	Mat resized;
	resize(roi_img, resized, Size(new_width, new_height));
	// 3.����Ŀ���С�ĺ�ɫ����
	Mat result = Mat::zeros(Size(target_size, target_size), roi_img.type());
	// 4.����ƫ���������з���
	int x_offset = (target_size - new_width) / 2;
	int y_offset = (target_size - new_height) / 2;
	// 5.�����ź��ͼ���Ƶ������м�
	Rect roi_rect(x_offset, y_offset, new_width, new_height);
	resized.copyTo(result(roi_rect));
	return result;
}
