#pragma once
#include "skel.h"
#include <sstream>
#include <numeric>
#include <algorithm>
#include <Eigen\Eigen>

#define FRAME_SIZE 500

class Person3DMotion
{
public:
	Person3DMotion();
	Person3D personInFrames[FRAME_SIZE];
	bool inViewFlag[FRAME_SIZE];
	std::vector<Eigen::VectorXd> jointsLocation[FRAME_SIZE];
	std::vector<Eigen::VectorXd> jointsAcceleration[FRAME_SIZE];
	std::vector<Eigen::VectorXd> jointsVelocity[FRAME_SIZE];
	void CalculateNewFrameMotion(int frameIdx, Person3D person);
};
