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

// CIOU���ڴ洢ƥ����
struct MatchResult {
	int candidateIndex;
	float ciou;
	bool isValid;
};

// �����ȶ�������
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
	Mat srcImg; // Դͼ��
	Mat binaryImg; // ��ֵ��ͼ��

	vector<BBox> last_lightBars; // �洢��һ֡�ĵ���BBox
	bool isInitialized; // ����Ƿ��ѳ�ʼ��
	float ciou_threshold; // CIOUƥ����ֵ

	vector<LightBar> lightBars; // ��⵽������
	vector<ArmorBox> armorBoxes;  // �洢��⵽��װ�װ�
	ArmorNumClassifier classifier; // װ�װ����ַ�����
	SimpleTimer timer; // ��ʱ��

	vector<LightBarStabilizer> lightbar_stabilizers; // ÿ�����Ƶ��ȶ�����
	bool stabilizer_initialized; // �ȶ������Ƿ��ѳ�ʼ��
public:
	ArmorDetected() = default;
	~ArmorDetected() = default;
	// ���װ�װ壨�����̺�����
	bool detectArmor(const Mat& src);
	// ͼ��Ԥ����
	void preprocessImage(const Mat& src);
	// ��ͳ�����������
	void findLightBarsTraditional();
	// �������
	void findLightBars();
	// װ�װ�ƥ��
	void matchArmors(); 
	// ������ʾ
	void showLightBars(Mat& debugImg) const;
	// ��ʾװ�װ�
	void showArmors(Mat& debugImg) const;
	// PnP����
	void solvePnPForAllArmors();
	// ��ʼ����ʼ֡�ĵ���
	void initializeLastLightBars(const Rect& l_roi, const Rect& r_roi);
	// ����ƥ�亯����CIOU��
	MatchResult findBestMatch(const vector<BBox>& candidates, const BBox& reference);
	// BBox��LightBar��ת��
	LightBar createLightBarFromBBox(const BBox& bbox);
	
	// ��ȡ�����
	const vector<LightBar>& getLightBars() const
	{
		return lightBars;
	}
	const vector<ArmorBox>& getArmorBoxes() const 
	{ return armorBoxes; 
	}
};