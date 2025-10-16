#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "LightBar.hpp"
#include "ArmorBox.hpp"
#include "ArmorNumClassifier.hpp"
#include "config.hpp"
#include "BBox.hpp"
#include <deque>  

using namespace cv;
using namespace std;

// CIOU用于存储匹配结果
struct MatchResult {
	int candidateIndex;
	float ciou;
	bool isValid;
};

// 条灯稳定化器类
class LightBarStabilizer {
private:
	deque<RotatedRect> history_rects;

public:
	RotatedRect stabilize(const RotatedRect& current_rect);
	void reset();
};

class ArmorDetected
{
private:
	Mat srcImg; // 源图像
	Mat binaryImg; // 二值化图像

	vector<BBox> last_lightBars; // 存储上一帧的灯条BBox
	bool isInitialized; // 标记是否已初始化
	float ciou_threshold; // CIOU匹配阈值

	vector<LightBar> lightBars; // 检测到的条灯
	vector<ArmorBox> armorBoxes;  // 存储检测到的装甲板
	ArmorNumClassifier classifier; // 装甲板数字分类器
	SimpleTimer timer; // 计时器

	vector<LightBarStabilizer> lightbar_stabilizers; // 每个条灯的稳定化器
	bool stabilizer_initialized; // 稳定化器是否已初始化
public:
	ArmorDetected() = default;
	~ArmorDetected() = default;
	// 检测装甲板（主流程函数）
	bool detectArmor(const Mat& src);
	// 图像预处理
	void preprocessImage(const Mat& src);
	// 传统方法检测条灯
	void findLightBarsTraditional();
	// 检测条灯
	void findLightBars();
	// 装甲板匹配
	void matchArmors(); 
	// 调试显示
	void showLightBars(Mat& debugImg) const;
	// 显示装甲板
	void showArmors(Mat& debugImg) const;
	// PnP解算
	void solvePnPForAllArmors();
	// 初始化起始帧的灯条
	void initializeLastLightBars(const Rect& l_roi, const Rect& r_roi);
	// 最优匹配函数（CIOU）
	MatchResult findBestMatch(const vector<BBox>& candidates, const BBox& reference);
	// BBox到LightBar的转换
	LightBar createLightBarFromBBox(const BBox& bbox);
	
	// 获取检测结果
	const vector<LightBar>& getLightBars() const
	{
		return lightBars;
	}
	const vector<ArmorBox>& getArmorBoxes() const 
	{ return armorBoxes; 
	}
};