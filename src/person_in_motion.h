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
	static const double PEOPLE_DISP_THRES;
	static const double JOINT_REBUILD_THRES;

	Person3DMotion();
	void RebuildPersonSkels(int frameIdx);
	void dfsJoints(int frameIdx, int thisJoint);
	Person3D personInFrames[FRAME_SIZE]; 
	Person2D ProjSkel(int frameIdx, const Eigen::Matrix<float, 3, 4>& proj) const;
	Person2D ProjSkelFineLocation(int frameIdx, const Eigen::Matrix<float, 3, 4>& proj) const;
	bool inViewFlag[FRAME_SIZE];
	std::vector<Eigen::VectorXd> jointsFineLocation[FRAME_SIZE];
	std::vector<Eigen::VectorXd> jointsLocation[FRAME_SIZE];
	std::vector<Eigen::VectorXd> jointsAcceleration[FRAME_SIZE];
	std::vector<Eigen::VectorXd> jointsVelocity[FRAME_SIZE];
	std::vector<double> jointsCredit[FRAME_SIZE];
	std::vector<int> rebuildStatus[FRAME_SIZE];
	std::vector<int> jointsStatus[FRAME_SIZE]; //0是正常，1需要校准，2错误
	//该测量点的可靠程度，为18个点的累加。测量到的点越多，与上次预测值的误差越少，则可靠程度值越高
	double totalCredits[FRAME_SIZE];
	//总可靠程度
	void CalculateNewFrameMotion(int frameIdx, Person3D person);
	void CalculateConstraintsSkels(int frameIdx, Person3D person);
	std::vector<int> const_skels_A, const_skels_B; //长度不变的骨骼作为物理约束，骨骼两端点为A，B
	std::vector<double> const_length;
};

