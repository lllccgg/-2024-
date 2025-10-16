/*
˼·��
��1����������ʹ���Լ�������ͷ������Ų�ͬ�Ƕȡ���ͬλ�õı궨��ͼƬ�����浽һ���ļ����У�Ȼ�����ı���¼ͼƬ·�����ڴ�����ͨ���ı�ȥ��ȡ·����
��2��ȷ���궨��Ĺ�񣬱����ڽǵ������7x10����ÿ�������ʵ�ʳߴ磨15mmx15mm����
��3����ͼƬ����Ԥ��������תΪ�Ҷ�ͼ����ֵ�˲�ȥ����������ͷ�ĵ�ͼƬ�����Խ�����������Ȼ��ʹ��findChessboardCorners������̸�ǵ㣬��ʹ��find4QuadCornerSubpix���������ؾ�����
��4��calibrateCameraȥ�궨���õ�����ڲξ���ͻ���ϵ����
��5����󣬽��л���У����Ч���Աȣ���ԭͼ��У�����ͼ������ʾ������ai����һ�δ��������궨Ч����
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


	// �궨

	// 1.�������̸�ÿ���ڽǵ����ͼ���еĶ�ά����
	Size board_size = Size(7, 10); //����궨���ڽǵ���Ŀ���У��У�
	vector<vector<Point2f>> imgsPoints;
	for (int i = 0; i < imgs.size(); i++)
	{
		Mat img1 = imgs[i];
		if (img1.empty()) {
			cout << "���棺��" << i << "��ͼ���ȡʧ�ܣ�����" << endl;
			continue;
		}

		Mat gray;
		cvtColor(img1, gray, COLOR_BGR2GRAY);
		medianBlur(gray, gray, 5); // ʹ��5x5�˽�����ֵ�˲���ȥ����������

		vector<Point2f> img1_points;

		// ������̸�ǵ����Ƿ�ɹ�
		bool found = findChessboardCorners(gray, board_size, img1_points);
		if (found && img1_points.size() == board_size.width * board_size.height) {
			// ֻ�м��ɹ�ʱ�Ž��������ؾ���
			find4QuadCornerSubpix(gray, img1_points, Size(5, 5));
			imgsPoints.push_back(img1_points);
			cout << "��" << i << "��ͼ��ǵ���ɹ�" << endl;
		}
		else {
			cout << "���棺��" << i << "��ͼ��ǵ���ʧ�ܣ�����" << endl;
		}
	}

	// 2.�������̸�ÿ���ڽǵ�Ŀռ���ά����
	Size squareSize = Size(15, 15); //���̸�ÿ���������ʵ�ߴ磨mm��
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

	// ����Ƿ����㹻����Чͼ����б궨
	if (imgsPoints.size() < 3) {
		cout << "������Чͼ���������㣨������Ҫ3�ţ�����ǰ��Чͼ��������" << imgsPoints.size() << endl;
		return -1;
	}

	// ͼ��ߴ�
	Size imgSize;
	imgSize.height = imgs[0].rows;
	imgSize.width = imgs[0].cols;
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); // ����ڲξ���
	Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0)); // ����� 5 ������ϵ���� k1,k2,p1,p2,k3
	vector<Mat> rvecs; //ÿ��ͼ�����ת����
	vector<Mat> tvecs; //ÿ��ͼ���ƽ������
	calibrateCamera(objectPoints, imgsPoints, imgSize, cameraMatrix, distCoeffs, rvecs, tvecs);

	cout << "������ڲξ���=" << endl << cameraMatrix << endl;
	cout << "�������ϵ��" << distCoeffs << endl;

	// 3. ����У����Ч���Ա�
	Mat original_image = imgs[0].clone();
	Mat undistorted_image;
	undistort(original_image, undistorted_image, cameraMatrix, distCoeffs);

	// ��ԭͼ��У�����ͼ������ʾ
	Mat comparison_image;
	hconcat(original_image, undistorted_image, comparison_image);
	imshow("Original vs Undistorted", comparison_image);


	waitKey(0);

	return 0;
}

/*
imgwidth=640
imghight=480

 ��һ��
������ڲξ���=
[602.1739431382024, 0, 324.5161999242971;
 0, 602.2550353079832, 243.9670856468874;
 0, 0, 1]
�������ϵ��[-0.3032852964679878, 1.137206167802484, 0.009141797230885046, -0.005460796172210933, -2.563687515095793]

�ڶ���
������ڲξ���=
[614.8391420674567, 0, 328.2661440947077;
 0, 615.1902697902865, 242.8073226730833;
 0, 0, 1]
�������ϵ��[-0.3311619176641216, 2.268420739505414, 0.001406899944638725, -0.0004803947135656995, -6.486810944575915]

������(����һ���궨��)
������ڲξ���=
[457.9902610081367, 0, 321.2587806755877;
 0, 458.5579838313615, 180.8540450017003;
 0, 0, 1]
�������ϵ��[-0.1562129821428429, 0.5707945090398995, 0.002475480560508443, -0.002766803551451849, -0.9785146669571626]

���Ĵ�
������ڲξ���=
[619.2342625843423, 0, 332.4107997008066;
 0, 619.4356974250945, 237.9417298019534;
 0, 0, 1]
�������ϵ��[-0.1857965552869972, 0.7150658975139499, 0.003183596780102963, 0.0009642865213194783, -0.8848275920367117]
*/