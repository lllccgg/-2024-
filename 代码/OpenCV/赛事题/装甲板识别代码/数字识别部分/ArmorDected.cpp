#include "ArmorDetected.hpp"

void eraseErrorRepeatArmor(vector<ArmorBox>& armors); // 用于删除游离灯条导致的错误装甲板

void ArmorDetected::preprocessImage(const Mat& src)
{
	GaussianBlur(srcImg, srcImg, Size(3, 3), 0, 0); // 低通滤波，减少条灯二值图的断裂

	Mat hsv;
	Mat blueChannel;

	cvtColor(srcImg, hsv, COLOR_BGR2HSV);
	inRange(hsv, Config::lower_blue, Config::upper_blue, blueChannel);

	// 2.二值化处理
	threshold(blueChannel, binaryImg, Config::BINARY_THRESHOLD, 255, THRESH_BINARY);

	// 3.形态学处理
	//morphologyEx(binaryImg, binaryImg, MORPH_CLOSE, Config::kernel_line); 
	Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 15));
	morphologyEx(binaryImg, binaryImg, MORPH_CLOSE, kernel);

	namedWindow("binary", WINDOW_GUI_NORMAL);
	imshow("binary", binaryImg);
}

void ArmorDetected::findLightBars()
{
	// 清除上一次检测到的条灯
	lightBars.clear();

	// 如果未初始化起始帧的条灯，使用传统检测方法
	if (!isInitialized || last_lightBars.size() < 2) 
	{
		findLightBarsTraditional(); // 传统轮廓检测
		return;
	}
	// 1.使用CIOU方法基于上一帧的条灯位置进行检测
	vector<BBox> candidateBoxes;
	vector<vector<Point>> contours;
	findContours(binaryImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	for (const auto& contour : contours) 
	{
		// 2.面积和点数筛选
		if (contourArea(contour) < Config::MIN_LIGHT_AREA ||
			contourArea(contour) > Config::MAX_LIGHT_AREA ||
			contour.size() < 5) continue;

		RotatedRect rotatedRect = fitEllipse(contour);
		LightBar lightBar(rotatedRect);

		if (lightBar.isValidLightBar()) 
		{
			BBox candidateBox(rotatedRect.boundingRect());
			candidateBoxes.push_back(candidateBox);
		}
	}
	// 3.基于CIOU的最优匹配
	MatchResult leftMatch = findBestMatch(candidateBoxes, last_lightBars[0]);
	MatchResult rightMatch = findBestMatch(candidateBoxes, last_lightBars[1]);
	// 避免重复匹配同一个候选框
	if (leftMatch.isValid && rightMatch.isValid &&
		leftMatch.candidateIndex == rightMatch.candidateIndex)
	{
		// 选择CIOU更高的匹配，另一个设为无效
		if (leftMatch.ciou > rightMatch.ciou)
		{
			rightMatch.isValid = false;
		}
		else
		{
			leftMatch.isValid = false;
		}
	}
	// 4.结果更新
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

	// 4. 更新last_lightBars（关键步骤）
	if (newLightBars.size() >= 2) 
	{
		last_lightBars = newLightBars; // 成功匹配，更新参考
		lightBars = detectedLightBars;
	}
	else
	{
		findLightBarsTraditional(); // 匹配失败，退回传统检测
	}
	// 在检测到条灯后，应用稳定化处理
	if (!stabilizer_initialized) {
		lightbar_stabilizers.resize(lightBars.size());
		stabilizer_initialized = true;
	}

	// 确保稳定化器数量与条灯数量匹配
	if (lightbar_stabilizers.size() != lightBars.size()) {
		lightbar_stabilizers.resize(lightBars.size());
	}

	// 对每个条灯应用稳定化
	for (int i = 0; i < lightBars.size(); i++) {
		RotatedRect stabilized_rect = lightbar_stabilizers[i].stabilize(lightBars[i].rect);
		lightBars[i] = LightBar(stabilized_rect);  // 用稳定化后的矩形重新创建条灯
	}
}

void ArmorDetected::showLightBars(Mat& debugImg) const
{
	// 在源图像上绘制检测到的条灯
	for (auto const& lightbar : lightBars)
	{
		Point2f vertices[4];
		lightbar.getVertices(vertices);
		// 绘制条灯轮廓
		for (int i = 0; i < 4; i++)
		{
			line(debugImg, vertices[i], vertices[(i + 1) % 4], Config::COLOR_GREEN, 2);
		}
		// 绘制中心点
		circle(debugImg, lightbar.center, 3, Config::COLOR_RED, -1);
		// 显示条灯信息（使用标准化角度）
		float normalizedAngle = abs(lightbar.angle);
		if (normalizedAngle > 45) normalizedAngle = 90 - normalizedAngle;
		string info = "L:" + to_string((int)lightbar.length) + "A:" + to_string((int)normalizedAngle);
		putText(debugImg, info, lightbar.center + Point2f(10, -10), FONT_HERSHEY_SIMPLEX, 0.4, Config::COLOR_YELLOW, 1);
	}
}

void ArmorDetected::matchArmors()
{
	// 清空上一次装甲板的检测结果
	armorBoxes.clear();
	// 至少需要2个条灯才能组成装甲板
	if (lightBars.size() < 2)
	{
		return;
	}
	// 双重循环遍历所有灯条组合
	for (size_t i = 0; i < lightBars.size() - 1; i++)
	{
		for (size_t j = 1; j < lightBars.size(); j++)
		{
			// 创建装甲板候选
			ArmorBox armorCandidate(lightBars[i], lightBars[j]);
			// 验证是否为有效的装甲板
			if (armorCandidate.isSuitableArmor())
			{
				// 使用LeNet-5进行数字识别
				classifier.loadONNXModel(Config::onnx_path); // 加载ONNX模型
				bool is_img = classifier.getArmorImg(armorCandidate, srcImg); // 获取装甲板图像
				if (is_img)
				{
					classifier.getArmorNumByONNX(armorCandidate); // 识别装甲板数字
					cout << "对装甲板进行数字识别" << endl;
				}
				armorBoxes.push_back(armorCandidate);
			}
		}
		eraseErrorRepeatArmor(armorBoxes); // 删除错误装甲板
	}
}

void ArmorDetected::showArmors(Mat& debugImg) const
{
	for (const auto& armorbox : armorBoxes)
	{
		// 绘制装甲板矩形框
		Point2f vertices[4];
		armorbox.getVertices(vertices);
		for (int i = 0; i < 4; i++)
		{
			line(debugImg, vertices[i], vertices[(i + 1) % 4], Config::COLOR_BLUE, 3);
		}
		// 绘制装甲板中心点
		circle(debugImg, armorbox.center, 5, Config::COLOR_RED, -1);
		// 显示装甲板信息
		string info = "Armor W:" + to_string((int)armorbox.width) +" H:" + to_string((int)armorbox.height);
		putText(debugImg, info, armorbox.center + Point2f(10, - 10), FONT_HERSHEY_SIMPLEX, 0.5, Config::COLOR_CYAN);
		string distText = "distance:" + to_string(armorbox.distance);
		putText(debugImg, distText, armorbox.center + Point2f(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
		string numText = "Num:" + to_string(armorbox.armorNum);
		putText(debugImg, numText, Point2f(10, 50), FONT_HERSHEY_SIMPLEX, 0.8, Config::COLOR_GREEN);
	}
}

bool ArmorDetected::detectArmor(const Mat& src)
{
	cout << "=== 开始装甲板检测 ===" << std::endl;
	// 保存源图像
	srcImg = src.clone();

	// 1. 图像预处理
	timer.start();
	preprocessImage(src);
	timer.printElapsed("图像预处理");

	// 2. 灯条检测
	timer.start();
	findLightBars();
	timer.printElapsed("条灯检测");

	// 3. 装甲板匹配及数字识别
	timer.start();
	matchArmors();
	timer.printElapsed("装甲板匹配及数字识别");
	
	// 4.PnP距离解算
	timer.start();
	solvePnPForAllArmors();
	timer.printElapsed("PnP距离解算");

	cout << "检测到装甲板数量: " << armorBoxes.size() << std::endl;
	cout << "=== 检测完成 ===" << std::endl << std::endl;
	
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

	// 从后往前删除，避免索引变化
	for (int i = armors.size() - 1; i >= 0; i--) {
		if (toDelete[i]) {
			armors.erase(armors.begin() + i);
		}
	}
}

void ArmorDetected::initializeLastLightBars(const Rect& l_roi, const Rect& r_roi)
{
	last_lightBars.clear();
	last_lightBars.push_back(BBox(l_roi, 0)); // 左灯条，class_id=0
	last_lightBars.push_back(BBox(r_roi, 1)); // 右灯条，class_id=1
	isInitialized = true;
}

void ArmorDetected::findLightBarsTraditional()
{
	lightBars.clear();
	// 1.轮廓检测
	vector<vector<Point>> contours;
	findContours(binaryImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	// 2.遍历轮廓，筛选条灯
	for (const auto& contour : contours)
	{
		double area = contourArea(contour);
		// 面积筛选
		if (area < Config::MIN_LIGHT_AREA || area > Config::MAX_LIGHT_AREA) continue;
		// 拟合椭圆（至少需要6个点才能拟合为椭圆）
		if (contour.size() < 6) continue;
		RotatedRect rotatedrect = fitEllipse(contour);
		// 创建条灯对象
		LightBar lightBar(rotatedrect);
		// 验证是否为有效条灯
		if (lightBar.isValidLightBar())
		{
			lightBars.push_back(lightBar);
		}
	}
	// 3. 重置跟踪状态
	isInitialized = false;
	last_lightBars.clear();

	// 4. 如果找到足够的灯条，重新初始化跟踪
	if (lightBars.size() >= 2) 
	{
		// 按x坐标排序，选择最左和最右的作为新的跟踪目标
		sort(lightBars.begin(), lightBars.end(),
			[](const LightBar& a, const LightBar& b) {
				return a.center.x < b.center.x;
			});

		// 重新初始化last_lightBars
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
	RotatedRect rotatedRect(center, size, 0); // 角度可以从历史信息推断
	return LightBar(rotatedRect);
}

// 稳定化器实现
RotatedRect LightBarStabilizer::stabilize(const RotatedRect& current_rect) {
	history_rects.push_back(current_rect);
	if (history_rects.size() > Config::STABILIZE_HISTORY_SIZE) {
		history_rects.pop_front();
	}

	// 计算加权平均
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
