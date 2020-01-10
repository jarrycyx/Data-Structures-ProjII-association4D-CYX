#pragma once
#include "skel.h"
#include <sstream>
#include <numeric>
#include <algorithm>
#include <Eigen\Eigen>
#include <vector>

#define FRAME_SIZE 500

class Person3DMotion
{
public:
	static const int ERR_THRES_REBUILD;
	static const double ERR_THRES_CALIBRT;
	static const double CREDT_THRES_REBUILD;
	static const double CREDT_THRES_REBUILD;
	static const double PEOPLE_DISP_THRES;

	Person3DMotion();
	Person3D personInFrames[FRAME_SIZE]; 
	Person2D ProjSkel(int frameIdx, const Eigen::Matrix<float, 3, 4>& proj) const;
	bool inViewFlag[FRAME_SIZE];
	std::vector<Eigen::VectorXd> jointsLocation[FRAME_SIZE];
	std::vector<Eigen::VectorXd> jointsAcceleration[FRAME_SIZE];
	std::vector<Eigen::VectorXd> jointsVelocity[FRAME_SIZE];
	std::vector<double> jointsCredit[FRAME_SIZE];
	double totalCredits[FRAME_SIZE];
	//�ò�����Ŀɿ��̶ȣ�Ϊ18������ۼӡ��������ĵ�Խ�࣬���ϴ�Ԥ��ֵ�����Խ�٣���ɿ��̶�ֵԽ��
	void CalculateNewFrameMotion(int frameIdx, Person3D person);
	void CalculateConstraintsSkels(int frameIdx, Person3D person);
	std::vector<int> const_skels_A, const_skels_B; //���Ȳ���Ĺ�����Ϊ����Լ�����������˵�ΪA��B
	std::vector<double> const_length;
};

