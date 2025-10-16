/*
思路：
（1）首先是先使用自己的摄像头拍摄多张不同角度、不同位置的标定板图片，保存到一个文件夹中，然后用文本记录图片路径，在代码中通过文本去读取路径；
（2）确定标定板的规格，比如内角点个数（7x10）和每个方格的实际尺寸（15mmx15mm）；
（3）对图片进行预处理，比如转为灰度图和中值滤波去噪声（摄像头拍的图片有明显椒盐噪声），然后使用findChessboardCorners检测棋盘格角点，并使用find4QuadCornerSubpix进行亚像素精化；
（4）calibrateCamera去标定，得到相机内参矩阵和畸变系数；
（5）最后，进行畸变校正与效果对比，将原图和校正后的图像并排显示（利用ai给的一段代码评估标定效果）
*/


#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;

int main()
{
	system("color F0");
	vector<Mat> imgs;
	string imageName;
	ifstream fin("D:\\Opencvpracticedata\\steroCalibDataL.txt");
	while (getline(fin, imageName))
	{
		Mat img = imread(imageName);
		imgs.push_back(img);
	}


	// 标定

	// 1.生成棋盘格每个内角点的在图像中的二维坐标
	Size board_size = Size(7, 10); //方格标定板内角点数目（行，列）
	vector<vector<Point2f>> imgsPoints;
	for (int i = 0; i < imgs.size(); i++)
	{
		Mat img1 = imgs[i];
		if (img1.empty()) {
			cout << "警告：第" << i << "张图像读取失败，跳过" << endl;
			continue;
		}

		Mat gray;
		cvtColor(img1, gray, COLOR_BGR2GRAY);
		medianBlur(gray, gray, 5); // 使用5x5核进行中值滤波以去除椒盐噪声

		vector<Point2f> img1_points;

		// 检查棋盘格角点检测是否成功
		bool found = findChessboardCorners(gray, board_size, img1_points);
		if (found && img1_points.size() == board_size.width * board_size.height) {
			// 只有检测成功时才进行亚像素精化
			find4QuadCornerSubpix(gray, img1_points, Size(5, 5));
			imgsPoints.push_back(img1_points);
			cout << "第" << i << "张图像角点检测成功" << endl;
		}
		else {
			cout << "警告：第" << i << "张图像角点检测失败，跳过" << endl;
		}
	}

	// 2.生成棋盘格每个内角点的空间三维坐标
	Size squareSize = Size(15, 15); //棋盘格每个方格的真实尺寸（mm）
	vector<vector<Point3f>> objectPoints;
	for (int i = 0; i < imgsPoints.size(); i++)
	{
		vector<Point3f> tempPointSet;
		for (int k = 0; k < board_size.height; k++)
		{
			for (int j = 0; j < board_size.width; j++)
			{
				Point3f realPoint;
				realPoint.x = j * squareSize.width;
				realPoint.y = k * squareSize.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		objectPoints.push_back(tempPointSet);
	}

	// 检查是否有足够的有效图像进行标定
	if (imgsPoints.size() < 3) {
		cout << "错误：有效图像数量不足（至少需要3张），当前有效图像数量：" << imgsPoints.size() << endl;
		return -1;
	}

	// 图像尺寸
	Size imgSize;
	imgSize.height = imgs[0].rows;
	imgSize.width = imgs[0].cols;
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); // 相机内参矩阵
	Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0)); // 相机的 5 个畸变系数： k1,k2,p1,p2,k3
	vector<Mat> rvecs; //每幅图像的旋转向量
	vector<Mat> tvecs; //每幅图像的平移向量
	calibrateCamera(objectPoints, imgsPoints, imgSize, cameraMatrix, distCoeffs, rvecs, tvecs);

	cout << "相机的内参矩阵=" << endl << cameraMatrix << endl;
	cout << "相机畸变系数" << distCoeffs << endl;

	// 3. 畸变校正与效果对比
	Mat original_image = imgs[0].clone();
	Mat undistorted_image;
	undistort(original_image, undistorted_image, cameraMatrix, distCoeffs);

	// 将原图和校正后的图像并排显示
	Mat comparison_image;
	hconcat(original_image, undistorted_image, comparison_image);
	imshow("Original vs Undistorted", comparison_image);


	waitKey(0);

	return 0;
}

/*
imgwidth=640
imghight=480

 第一次
相机的内参矩阵=
[602.1739431382024, 0, 324.5161999242971;
 0, 602.2550353079832, 243.9670856468874;
 0, 0, 1]
相机畸变系数[-0.3032852964679878, 1.137206167802484, 0.009141797230885046, -0.005460796172210933, -2.563687515095793]

第二次
相机的内参矩阵=
[614.8391420674567, 0, 328.2661440947077;
 0, 615.1902697902865, 242.8073226730833;
 0, 0, 1]
相机畸变系数[-0.3311619176641216, 2.268420739505414, 0.001406899944638725, -0.0004803947135656995, -6.486810944575915]

第三次(用另一个标定板)
相机的内参矩阵=
[457.9902610081367, 0, 321.2587806755877;
 0, 458.5579838313615, 180.8540450017003;
 0, 0, 1]
相机畸变系数[-0.1562129821428429, 0.5707945090398995, 0.002475480560508443, -0.002766803551451849, -0.9785146669571626]

第四次
相机的内参矩阵=
[619.2342625843423, 0, 332.4107997008066;
 0, 619.4356974250945, 237.9417298019534;
 0, 0, 1]
相机畸变系数[-0.1857965552869972, 0.7150658975139499, 0.003183596780102963, 0.0009642865213194783, -0.8848275920367117]
*/