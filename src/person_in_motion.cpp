#include "person_in_motion.h"
#include "skel.h"
#include <iostream>

Person3DMotion::Person3DMotion()
{
	for (int i = 0; i < FRAME_SIZE; i++)
	{
		inViewFlag[i] = false;
		jointsAcceleration[i].resize(GetSkelDef().jointSize);
		jointsVelocity[i].resize(GetSkelDef().jointSize);
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
		{
			jointsAcceleration[i][jIdx] = Eigen::VectorXd::Constant(3, 0);
			jointsVelocity[i][jIdx] = Eigen::VectorXd::Constant(3, 0);
		}
	}
}


void Person3DMotion::CalculateNewFrameMotion(int frameIdx, Person3D person)
{
	personInFrames[frameIdx] = person;
	inViewFlag[frameIdx] = true;
	for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
	{
		Eigen::VectorXd newLoc;
		newLoc.resize(3);
		jointsLocation[frameIdx].push_back(newLoc);
		jointsLocation[frameIdx][jIdx][0] = personInFrames[frameIdx].joints(0, jIdx);
		jointsLocation[frameIdx][jIdx][1] = personInFrames[frameIdx].joints(1, jIdx);
		jointsLocation[frameIdx][jIdx][2] = personInFrames[frameIdx].joints(2, jIdx);
		if (frameIdx != 0 && inViewFlag[frameIdx - 1])
		{
			Eigen::VectorXd r2 = jointsLocation[frameIdx][jIdx];
			Eigen::VectorXd r1 = jointsLocation[frameIdx - 1][jIdx];
			jointsVelocity[frameIdx][jIdx] = r2 - r1;
		}
		if (frameIdx > 1 && inViewFlag[frameIdx - 1])
		{
			Eigen::VectorXd v2 = jointsVelocity[frameIdx][jIdx];
			Eigen::VectorXd v1 = jointsVelocity[frameIdx - 1][jIdx];
			jointsAcceleration[frameIdx][jIdx] = v2 - v1;
		}
	}
}