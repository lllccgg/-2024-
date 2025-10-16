#include "ArmorNumClassifier.hpp"

// 透视变换辅助函数
void WarpedImage(const cv::Mat& src, cv::Mat& dst, const cv::Point2f srcPoints[4], const cv::Size& dstSize);
// 保持长宽比的智能填充函数
Mat preprocessROI(const Mat& roi_img, int target_size);

ArmorNumClassifier::ArmorNumClassifier()
{
	armorImgSize = Size(32, 32); // 输入图像尺寸
	use_onnx = false; // 是否加载ONNX模型
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
				cout << "ONNX模型加载失败!" << endl;
				exit(0);
			}
			// 使用CPU
			onnx_net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
			onnx_net.setPreferableTarget(dnn::DNN_TARGET_CPU);
			use_onnx = true;
			cout << "ONNX模型加载成功!" << endl;
			//waitKey(0); // 测试 
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
	// 计算ROI区域的坐标
	int x = max(0, (int)armor.vertices[0].x + Config::roi_x);
	int y = max(0, (int)armor.vertices[0].y - Config::roi_y);
	int width = min((int)armor.width - Config::x_offset, srcImg.cols - x);
	int height = min((int)armor.height + Config::y_offset, srcImg.rows - y);

	// 检查ROI区域是否有效
	if (width <= 0 || height <= 0) 
	{
		cout << "ROI区域的宽度或高度无效: width=" << width << ", height=" << height << endl;
		return false;
	}
	// 添加最小尺寸要求
	const int MIN_ROI_SIZE = 5; // 最小ROI尺寸
	if (width < MIN_ROI_SIZE || height < MIN_ROI_SIZE) 
	{
		cout << "ROI区域太小，无法处理: width=" << width << ", height=" << height
			<< " (最小要求: " << MIN_ROI_SIZE << ")" << endl;
		armor.armorNum = -1;
		return false;
	}

	if (x  < 0 || y  < 0 ||
		width  > srcImg.cols ||
		height  > srcImg.rows) 
	{
		cout << "ROI区域超出了图像边界" << endl;
		armor.armorNum = -1; // 标记为无效装甲板
		return false;
	}

	if (x + width > srcImg.cols || y + height > srcImg.rows) 
	{
		cout << "ROI区域超出图像边界: x=" << x << ", y=" << y
			<< ", width=" << width << ", height=" << height
			<< ", img_cols=" << srcImg.cols << ", img_rows=" << srcImg.rows << endl;
		armor.armorNum = -1;
		return false;
	}

	Mat roi;
	roi = srcImg(Rect(x, y, width, height)); // 提取ROI区域

	// 添加ROI有效性检查
	if (roi.empty() || roi.cols <= 0 || roi.rows <= 0) 
	{
		cout << "提取的ROI区域无效或为空" << endl;
		armor.armorNum = -1;
		return false;
	}

	for (int i = 0; i < 4; i++)
	{
		srcPoints[i] = armor.vertices[i] ; // 装甲板顶点坐标
	}

	warpPerspective_dst = preprocessROI(roi, 32); // 智能填充

	if (!warpPerspective_dst.empty() &&
		warpPerspective_dst.cols > 0 &&
		warpPerspective_dst.rows > 0) 
	{
		namedWindow("Warped Image", WINDOW_GUI_NORMAL);
		imshow("Warped Image", warpPerspective_dst);
	}
	else 
	{
		cout << "警告：处理后的图像无效，跳过显示" << endl;
	}

	namedWindow("Warped Image", WINDOW_GUI_NORMAL);
	imshow("Warped Image", warpPerspective_dst);
	//imwrite("D:\\Opencvpracticedata\\warp0.png", warpPerspective_dst);
	return true;
}

void ArmorNumClassifier::getArmorNumByONNX(ArmorBox& armor)
{
	// 检查模型是否加载成功
	if (!use_onnx) {
		cout << "ONNX模型未加载!" << endl;
		return;
	}

	// 1. 对提取的ROI区域进行预处理
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

	// 2.规范图像大小
	Mat resized_img;
	if (processedImg.size() != armorImgSize)
	{
		resize(processedImg, resized_img, armorImgSize);
	}
	else
	{
		resized_img = processedImg;
	}
	
	// 3.对图像进行二值化
	Mat binary;
	threshold(resized_img, binary, Config::bianry, 255, THRESH_BINARY_INV);

	// 4.形态学操作
	//Mat kernel = getStructuringElement(MORPH_RECT, Size(1, 2));
	//morphologyEx(binary, binary, MORPH_OPEN, kernel);


	namedWindow("ONNX Input", WINDOW_GUI_NORMAL);
	imshow("ONNX Input", binary);
	//waitKey(0);
	
	double maxval, minval;
	Point maxloc, minloc;
	minMaxLoc(binary, &minval, &maxval, &minloc, &maxloc);
	//cout << "归一化前：" << " minval=" << minval << ", maxval=" << maxval << endl;

	// 7.归一化
	binary.convertTo(binary, CV_32F, 1.0 / 255.0);
	
	minMaxLoc(binary, &minval, &maxval, &minloc, &maxloc);
	//cout << "归一化后：" << " minval=" << minval << ", maxval=" << maxval << endl;

	// 8. 创建4D blob
	Mat blob = dnn::blobFromImage(binary, 1.0, armorImgSize, Scalar(0), false, false);
	//cout << "N=" << blob.size[0] << ", C=" << blob.size[1] << ", H=" << blob.size[2] << ", W=" << blob.size[3] << endl;

	// 9.设置网络输入
	onnx_net.setInput(blob);

	// 10.前向传播，获取输出
	Mat output = onnx_net.forward();



	// 11. 获取预测结果
	Point classIdPoint;
	double confidence;
	minMaxLoc(output, nullptr, &confidence, nullptr, &classIdPoint);
	armor.armorNum = classIdPoint.x; // 类别索引即为预测的数字
}

// 透视变换辅助函数
void WarpedImage(const cv::Mat& src, cv::Mat& dst, const cv::Point2f srcPoints[4], const cv::Size& dstSize)
{
	CV_Assert(!src.empty());
	//std::cout << "原图大小: " << src.cols << "x" << src.rows << "  类型: " << src.type() << std::endl;

	// 不修改原图：生成浮点副本
	cv::Mat src_float;
	src.convertTo(src_float, CV_32FC3, 1.0 / 255.0);

	// 打印输入角点
	//std::cout << "srcPoints:\n";
	//for (int i = 0; i < 4; ++i)
	//	std::cout << i << ": " << srcPoints[i] << std::endl;

	// 计算透视变换矩阵
	cv::Point2f dstPoints[4] = {
		{0, 0},
		{(float)dstSize.width - 1, 0},
		{(float)dstSize.width - 1, (float)dstSize.height - 1},
		{0, (float)dstSize.height - 1}
	};

	cv::Mat M = cv::getPerspectiveTransform(srcPoints, dstPoints);

	//std::cout << "变换矩阵 M:\n" << M << std::endl;

	// 执行透视变换
	cv::warpPerspective(src_float, dst, M, dstSize, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

	// 检查输出范围
	double minVal, maxVal;
	cv::minMaxLoc(dst.reshape(1), &minVal, &maxVal);
	//std::cout << "warp结果范围: " << minVal << " ~ " << maxVal << std::endl;

	// 乘回255并转为8位显示
	dst.convertTo(dst, CV_8UC3, 255.0);

	cv::imshow("warp result", dst);
	//cv::waitKey(0);
}

// 保持长宽比的智能填充函数
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

	// 1.计算缩放比例，保持长宽比
	double scale = min((double)target_size / roi_img.cols, (double)target_size / roi_img.rows);
	
	if (scale <= 0 || !isfinite(scale)) {
		std::cerr << "Error: invalid scale = " << scale << std::endl;
		return Mat();
	}

	// 2.按比例缩放
	int new_width = int(roi_img.cols * scale);
	int new_height = int(roi_img.rows * scale);

	// 确保最小尺寸为1像素
	new_width = max(1, new_width);
	new_height = max(1, new_height);

	// 添加调试信息
	//std::cout << "ROI原始尺寸: " << roi_img.cols << "x" << roi_img.rows
	//	<< ", 缩放比例: " << scale
	//	<< ", 新尺寸: " << new_width << "x" << new_height << std::endl;

	// 添加尺寸检查
	if (new_width <= 0 || new_height <= 0) {
		std::cerr << "Error: calculated new dimensions invalid! new_width="
			<< new_width << ", new_height=" << new_height << std::endl;
		return Mat();
	}

	Mat resized;
	resize(roi_img, resized, Size(new_width, new_height));
	// 3.创建目标大小的黑色背景
	Mat result = Mat::zeros(Size(target_size, target_size), roi_img.type());
	// 4.计算偏移量，居中放置
	int x_offset = (target_size - new_width) / 2;
	int y_offset = (target_size - new_height) / 2;
	// 5.将缩放后的图像复制到背景中间
	Rect roi_rect(x_offset, y_offset, new_width, new_height);
	resized.copyTo(result(roi_rect));
	return result;
}
