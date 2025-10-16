#include "ArmorDetected.hpp"


void eraseErrorRepeatArmor(vector<ArmorBox>& armors); // ����ɾ������������µĴ���װ�װ�

//ArmorDetected::ArmorDetected() : stabilizer_initialized(false) 
//{
//	
//}

void ArmorDetected::preprocessImage(const Mat& src)
{
	srcImg = src.clone();
	Mat hsv;
	Mat blueChannel;

	// 1.1 HSV�ռ����ɫ��ȡ
	cvtColor(srcImg, hsv, COLOR_BGR2HSV);
	inRange(hsv, Config::lower_blue, Config::upper_blue, blueChannel);

	// 2.��ֵ������
	threshold(blueChannel, binaryImg, Config::BINARY_THRESHOLD, 255, THRESH_BINARY);

	// 3.��̬ѧ����ȥ��
	morphologyEx(binaryImg, binaryImg, MORPH_CLOSE, Config::kernel_line); 

	namedWindow("binary", WINDOW_GUI_NORMAL);
	imshow("binary", binaryImg);
}

void ArmorDetected::findLightBars()
{
	// �����һ�μ�⵽������
	lightBars.clear();

	// ���δ��ʼ����ʼ֡�����ƣ�ʹ�ô�ͳ��ⷽ��
	if (!isInitialized || last_lightBars.size() < 2)
	{
		findLightBarsTraditional(); // ��ͳ�������
		return;
	}
	// 1.ʹ��CIOU����������һ֡������λ�ý��м��
	vector<BBox> candidateBoxes;
	vector<vector<Point>> contours;
	findContours(binaryImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	for (const auto& contour : contours)
	{
		// 2.����͵���ɸѡ
		if (contourArea(contour) < Config::MIN_LIGHT_AREA ||
			contourArea(contour) > Config::MAX_LIGHT_AREA ||
			contour.size() < 6) continue;

		RotatedRect rotatedRect = fitEllipse(contour);
		LightBar lightBar(rotatedRect);

		if (lightBar.isValidLightBar())
		{
			BBox candidateBox(rotatedRect.boundingRect());
			candidateBoxes.push_back(candidateBox);
		}
	}
	// 3.����CIOU������ƥ��
	MatchResult leftMatch = findBestMatch(candidateBoxes, last_lightBars[0]);
	MatchResult rightMatch = findBestMatch(candidateBoxes, last_lightBars[1]);
	// �����ظ�ƥ��ͬһ����ѡ��
	if (leftMatch.isValid && rightMatch.isValid &&
		leftMatch.candidateIndex == rightMatch.candidateIndex)
	{
		// ѡ��CIOU���ߵ�ƥ�䣬��һ����Ϊ��Ч
		if (leftMatch.ciou > rightMatch.ciou)
		{
			rightMatch.isValid = false;
		}
		else
		{
			leftMatch.isValid = false;
		}
	}
	// 4.�������
	vector<BBox> newLightBars;
	vector<LightBar> detectedLightBars;

	if (leftMatch.isValid)
	{
		BBox leftBox = candidateBoxes[leftMatch.candidateIndex];
		newLightBars.push_back(leftBox);
		detectedLightBars.push_back(createLightBarFromBBox(leftBox));
	}

	if (rightMatch.isValid)
	{
		BBox rightBox = candidateBoxes[rightMatch.candidateIndex];
		newLightBars.push_back(rightBox);
		detectedLightBars.push_back(createLightBarFromBBox(rightBox));
	}

	// 4. ����last_lightBars���ؼ����裩
	if (newLightBars.size() >= 2)
	{
		last_lightBars = newLightBars; // �ɹ�ƥ�䣬���²ο�
		lightBars = detectedLightBars;
	}
	else
	{
		findLightBarsTraditional(); // ƥ��ʧ�ܣ��˻ش�ͳ���
	}
	// �ڼ�⵽���ƺ�Ӧ���ȶ�������
	if (!stabilizer_initialized) {
		lightbar_stabilizers.resize(lightBars.size());
		stabilizer_initialized = true;
	}

	// ȷ���ȶ�������������������ƥ��
	if (lightbar_stabilizers.size() != lightBars.size()) {
		lightbar_stabilizers.resize(lightBars.size());
	}

	// ��ÿ������Ӧ���ȶ���
	for (int i = 0; i < lightBars.size(); i++) {
		RotatedRect stabilized_rect = lightbar_stabilizers[i].stabilize(lightBars[i].rect);
		lightBars[i] = LightBar(stabilized_rect);  // ���ȶ�����ľ������´�������
	}
}

void ArmorDetected::showLightBars(Mat& debugImg) const
{
	// ��Դͼ���ϻ��Ƽ�⵽������
	for (auto const& lightbar : lightBars)
	{
		Point2f vertices[4];
		lightbar.getVertices(vertices);
		// ������������
		for (int i = 0; i < 4; i++)
		{
			line(debugImg, vertices[i], vertices[(i + 1) % 4], Config::COLOR_GREEN, 2);
		}
		// �������ĵ�
		circle(debugImg, lightbar.center, 3, Config::COLOR_RED, -1);
		// ��ʾ������Ϣ��ʹ�ñ�׼���Ƕȣ�
		float normalizedAngle = abs(lightbar.angle);
		if (normalizedAngle > 45) normalizedAngle = 90 - normalizedAngle;
		string info = "L:" + to_string((int)lightbar.length) + "A:" + to_string((int)normalizedAngle);
		putText(debugImg, info, lightbar.center + Point2f(10, -10), FONT_HERSHEY_SIMPLEX, 0.4, Config::COLOR_YELLOW, 1);
	}
}

void ArmorDetected::matchArmors()
{
	// �����һ��װ�װ�ļ����
	armorBoxes.clear();
	// ������Ҫ2�����Ʋ������װ�װ�
	if (lightBars.size() < 2)
	{
		return;
	}
	// ˫��ѭ���������е������
	for (size_t i = 0; i < lightBars.size() - 1; i++)
	{
		for (size_t j = 1; j < lightBars.size(); j++)
		{
			// ����װ�װ��ѡ
			ArmorBox armorCandidate(lightBars[i], lightBars[j]);
			// ��֤�Ƿ�Ϊ��Ч��װ�װ�
			if (armorCandidate.isSuitableArmor())
			{
				armorBoxes.push_back(armorCandidate);
			}
		}
		eraseErrorRepeatArmor(armorBoxes); // ɾ������װ�װ�
	}
}

void ArmorDetected::showArmors(Mat& debugImg) const
{
	for (const auto& armorbox : armorBoxes)
	{
		// ����װ�װ���ο�
		Point2f vertices[4];
		armorbox.getVertices(vertices);
		for (int i = 0; i < 4; i++)
		{
			line(debugImg, vertices[i], vertices[(i + 1) % 4], Config::COLOR_BLUE, 3);
		}
		// ����װ�װ����ĵ�
		circle(debugImg, armorbox.center, 5, Config::COLOR_RED, -1);
		// ��ʾװ�װ���Ϣ
		string info = "Armor W:" + to_string((int)armorbox.width) + " H:" + to_string((int)armorbox.height);
		putText(debugImg, info, armorbox.center + Point2f(10, -10), FONT_HERSHEY_SIMPLEX, 0.5, Config::COLOR_CYAN);
		string distText = "distance:" + to_string(armorbox.distance);
		putText(debugImg, distText, armorbox.center + Point2f(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
		string numText = "Num:" + to_string(armorbox.armorNum);
		putText(debugImg, numText, Point2f(10, 50), FONT_HERSHEY_SIMPLEX, 0.8, Config::COLOR_GREEN);
	}
}

bool ArmorDetected::detectArmor(const Mat& src)
{
	cout << "=== ��ʼװ�װ��� ===" << std::endl;
	// ����Դͼ��
	srcImg = src.clone();

	// 1. ͼ��Ԥ����
	timer.start();
	preprocessImage(src);
	timer.printElapsed("ͼ��Ԥ����");

	// 2. �������
	timer.start();
	findLightBars();
	timer.printElapsed("���Ƽ��");

	// 3. װ�װ�ƥ��
	timer.start();
	matchArmors();
	timer.printElapsed("װ�װ�ƥ��");

	// 4.PnP�������
	timer.start();
	solvePnPForAllArmors();
	timer.printElapsed("PnP�������");

	cout << "��⵽װ�װ�����: " << armorBoxes.size() << std::endl;
	cout << "=== ������ ===" << std::endl << std::endl;

	return !armorBoxes.empty();
}

void ArmorDetected::solvePnPForAllArmors()
{
	for (auto& armorbox : armorBoxes)
	{
		armorbox.solvePnP(Config::cameraMatrix, Config::distCoeffs);
	}
}

void eraseErrorRepeatArmor(vector<ArmorBox>& armors)
{
	vector<bool> toDelete(armors.size(), false);

	for (size_t i = 0; i < armors.size(); i++)
	{
		if (toDelete[i]) continue;

		for (size_t j = i + 1; j < armors.size(); j++)
		{
			if (toDelete[j]) continue;

			if (armors[i].leftlight.center == armors[j].leftlight.center ||
				armors[i].leftlight.center == armors[j].rightlight.center ||
				armors[i].rightlight.center == armors[j].leftlight.center ||
				armors[i].rightlight.center == armors[j].rightlight.center)
			{
				if (armors[i].getDeviationAngle() > armors[j].getDeviationAngle()) {
					toDelete[i] = true;
					break;
				}
				else {
					toDelete[j] = true;
				}
			}
		}
	}

	// �Ӻ���ǰɾ�������������仯
	for (int i = armors.size() - 1; i >= 0; i--) {
		if (toDelete[i]) {
			armors.erase(armors.begin() + i);
		}
	}
}

void ArmorDetected::initializeLastLightBars(const Rect& l_roi, const Rect& r_roi)
{
	last_lightBars.clear();
	last_lightBars.push_back(BBox(l_roi, 0)); // �������class_id=0
	last_lightBars.push_back(BBox(r_roi, 1)); // �ҵ�����class_id=1
	isInitialized = true;
}

void ArmorDetected::findLightBarsTraditional()
{
	lightBars.clear();
	// 1.�������
	vector<vector<Point>> contours;
	findContours(binaryImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	// 2.����������ɸѡ����
	for (const auto& contour : contours)
	{
		double area = contourArea(contour);
		// ���ɸѡ
		if (area < Config::MIN_LIGHT_AREA || area > Config::MAX_LIGHT_AREA) continue;
		// �����Բ��������Ҫ6����������Ϊ��Բ��
		if (contour.size() < 6) continue;
		RotatedRect rotatedrect = fitEllipse(contour);
		// �������ƶ���
		LightBar lightBar(rotatedrect);
		// ��֤�Ƿ�Ϊ��Ч����
		if (lightBar.isValidLightBar())
		{
			lightBars.push_back(lightBar);
		}
	}
	// 3. ���ø���״̬
	isInitialized = false;
	last_lightBars.clear();

	// 4. ����ҵ��㹻�ĵ��������³�ʼ������
	if (lightBars.size() >= 2)
	{
		// ��x��������ѡ����������ҵ���Ϊ�µĸ���Ŀ��
		sort(lightBars.begin(), lightBars.end(),
			[](const LightBar& a, const LightBar& b) {
				return a.center.x < b.center.x;
			});

		// ���³�ʼ��last_lightBars
		last_lightBars.clear();
		last_lightBars.push_back(BBox(lightBars[0].rect.boundingRect()));
		last_lightBars.push_back(BBox(lightBars[lightBars.size() - 1].rect.boundingRect()));
		isInitialized = true;
	}
}

MatchResult ArmorDetected::findBestMatch(const vector<BBox>& candidates, const BBox& reference)
{
	MatchResult result = { -1, -1.0f, false };

	for (size_t i = 0; i < candidates.size(); i++)
	{
		float ciou = candidates[i].ciou(reference);
		if (ciou > result.ciou && ciou > ciou_threshold)
		{
			result.candidateIndex = i;
			result.ciou = ciou;
			result.isValid = true;
		}
	}
	return result;
}

LightBar ArmorDetected::createLightBarFromBBox(const BBox& bbox)
{
	Point2f center(bbox.x + bbox.width / 2.0f, bbox.y + bbox.height / 2.0f);
	Size2f size(bbox.width, bbox.height);
	RotatedRect rotatedRect(center, size, 0); // �Ƕȿ��Դ���ʷ��Ϣ�ƶ�
	return LightBar(rotatedRect);
}

// �ȶ�����ʵ��
RotatedRect LightBarStabilizer::stabilize(const RotatedRect& current_rect) {
	history_rects.push_back(current_rect);
	if (history_rects.size() > Config::STABILIZE_HISTORY_SIZE) {
		history_rects.pop_front();
	}

	// �����Ȩƽ��
	Point2f avg_center(0, 0);
	Size2f avg_size(0, 0);
	float avg_angle = 0;
	float total_weight = 0;

	for (int i = 0; i < history_rects.size(); i++) {
		float weight = (i == history_rects.size() - 1) ? Config::STABILIZE_WEIGHT :
			(1.0f - Config::STABILIZE_WEIGHT) / (history_rects.size() - 1);

		avg_center += history_rects[i].center * weight;
		avg_size.width += history_rects[i].size.width * weight;
		avg_size.height += history_rects[i].size.height * weight;
		avg_angle += history_rects[i].angle * weight;
		total_weight += weight;
	}

	return RotatedRect(avg_center, avg_size, avg_angle);
}

void LightBarStabilizer::reset() {
	history_rects.clear();
}