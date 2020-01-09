#include "person_in_motion.h"
#include "skel.h"
#include <iostream>

Person3DMotion::Person3DMotion()
{
	for (int i = 0; i < FRAME_SIZE; i++)
	{
		jointsAcceleration[i].resize(GetSkelDef().jointSize);
		jointsVelocity[i].resize(GetSkelDef().jointSize);
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
		{
			jointsAcceleration[i][jIdx] = Eigen::VectorXd::Constant(3, -1);
			jointsVelocity[i][jIdx] = Eigen::VectorXd::Constant(3, -1);
		}
	}
}


void Person3DMotion::CalculateNewFrameMotion(int frameIdx, Person3D person)
{
	personInFrames.emplace_back(person);
	for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
	{
		if (frameIdx != 0)
		{
			Eigen::VectorXd r1, r2;
			r1.resize(3);
			r2.resize(3);
			r2[0] = personInFrames[frameIdx].joints(0, jIdx);
			r2[1] = personInFrames[frameIdx].joints(1, jIdx);
			r2[2] = personInFrames[frameIdx].joints(2, jIdx);
			r2[0] = personInFrames[int(frameIdx) - 1].joints(0, jIdx);
			r2[1] = personInFrames[int(frameIdx) - 1].joints(1, jIdx);
			r2[2] = personInFrames[int(frameIdx) - 1].joints(2, jIdx);
			jointsVelocity[frameIdx][jIdx] = r2 - r1;
		}
		if (frameIdx > 1)
		{
			Eigen::VectorXd v2 = jointsVelocity[frameIdx][jIdx];
			Eigen::VectorXd v1 = jointsVelocity[frameIdx - 1][jIdx];
			jointsAcceleration[frameIdx][jIdx] = v2 - v1;
		}
	}
}