#include "person_in_motion.h"
#include "skel.h"
#include <iostream>


const int Person3DMotion::ERR_THRES_REBUILD = 1;
const double Person3DMotion::ERR_THRES_CALIBRT = 0.1;
const double Person3DMotion::CREDT_THRES_REBUILD = 4;
const double Person3DMotion::CREDT_THRES_REBUILD = 4;
const double Person3DMotion::PEOPLE_DISP_THRES = 5;


Person3DMotion::Person3DMotion()
{
	const_skels_A = {4, 4, 1, 1, 5, 6, 11, 12, 1, 0, 0, 2, 3, 7, 8, 13, 14};
	const_skels_B = {9, 10, 5, 6, 11, 12, 15, 16, 0, 2, 3, 7, 8, 13, 14, 18, 17};
	for (int i = 0; i < FRAME_SIZE; i++)
	{
		totalCredits[i] = 0;
		inViewFlag[i] = false;
		jointsAcceleration[i].resize(GetSkelDef().jointSize);
		jointsVelocity[i].resize(GetSkelDef().jointSize);
		jointsCredit[i].resize(GetSkelDef().jointSize);
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
		{
			jointsAcceleration[i][jIdx] = Eigen::VectorXd::Constant(3, 0);
			jointsVelocity[i][jIdx] = Eigen::VectorXd::Constant(3, 0);
			jointsCredit[i][jIdx] = 0;
		}
	}
}


//¼ÆËãÍ¶Ó°
Person2D Person3DMotion::ProjSkel(int frameIdx, const Eigen::Matrix<float, 3, 4>& proj) const {
	Person2D person;
	Eigen::Matrix4Xf joints;
	joints = Eigen::Matrix4Xf::Zero(4, GetSkelDef().jointSize);
	for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
	{
		joints(0, jIdx) = jointsLocation[frameIdx][jIdx][0];
		joints(1, jIdx) = jointsLocation[frameIdx][jIdx][1];
		joints(2, jIdx) = jointsLocation[frameIdx][jIdx][2];
		joints(3, jIdx) = (jointsLocation[frameIdx][jIdx][0] != 0
			&& jointsLocation[frameIdx][jIdx][1] != 0
			&& jointsLocation[frameIdx][jIdx][2] != 0);
	}

	person.joints.topRows(2) = (proj * (joints.topRows(3).colwise().homogeneous())).colwise().hnormalized();
	person.joints.row(2) = joints.row(3);
	for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++)
		person.pafs[pafIdx] = person.joints(GetSkelDef().pafDict(0, pafIdx)) > FLT_EPSILON
		&& person.joints(GetSkelDef().pafDict(1, pafIdx)) > FLT_EPSILON ? 1.f : 0.f;
	return person;
}

void Person3DMotion::CalculateConstraintsSkels(int frameIdx, Person3D person)
{
	int consSize = const_skels_A.size();
	for (int cIdx = 0; cIdx < consSize; cIdx++)
	{
		int jointA = const_skels_A[cIdx], jointB = const_skels_B[cIdx];
		Eigen::VectorXd delta = (jointsLocation[frameIdx][jointA] - jointsLocation[frameIdx][jointB]);
		const_length[cIdx] = delta.norm();

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
			if (jointsLocation[frameIdx - 1][jIdx][0] != 0 && jointsLocation[frameIdx - 1][jIdx][1] != 0 && jointsLocation[frameIdx - 1][jIdx][2] != 0)
				if (jointsLocation[frameIdx][jIdx][0] != 0 && jointsLocation[frameIdx][jIdx][1] != 0 && jointsLocation[frameIdx][jIdx][2] != 0)
					jointsVelocity[frameIdx][jIdx] = r2 - r1;
		}
		if (frameIdx > 1 && inViewFlag[frameIdx - 1])
		{
			Eigen::VectorXd v2 = jointsVelocity[frameIdx][jIdx];
			Eigen::VectorXd v1 = jointsVelocity[frameIdx - 1][jIdx];
			if (jointsVelocity[frameIdx - 1][jIdx][0] != 0 && jointsVelocity[frameIdx - 1][jIdx][0] != 0 && jointsVelocity[frameIdx - 1][jIdx][0] != 0)
				if (jointsLocation[frameIdx][jIdx][0] != 0 && jointsLocation[frameIdx][jIdx][1] != 0 && jointsLocation[frameIdx][jIdx][2] != 0)
					jointsAcceleration[frameIdx][jIdx] = v2 - v1;
		}
		
	}


	for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
	{
		if (frameIdx != 0 && inViewFlag[frameIdx - 1])
		{
			Eigen::VectorXd lastVec = jointsVelocity[frameIdx - 1][jIdx];
			Eigen::VectorXd lastAccel = jointsAcceleration[frameIdx - 1][jIdx];
			Eigen::VectorXd predLoc = jointsLocation[frameIdx - 1][jIdx] + lastVec + lastAccel;
			Eigen::VectorXd error = predLoc - jointsLocation[frameIdx][jIdx];
			if (jointsLocation[frameIdx - 1][jIdx][0] == 0)
				error = Eigen::VectorXd::Constant(3, 0.01);
			double jointProb = person.joints(3, jIdx);
			double totalError = abs(error[0]) + abs(error[1]) + abs(error[2]) + 0.03 / (0.01 + jointProb);
			if (totalError > ERR_THRES_REBUILD)
			{
				jointsCredit[frameIdx][jIdx] = jointsCredit[frameIdx - 1][jIdx] * 0.6;
			}
			else if (totalError > ERR_THRES_CALIBRT)
			{
				jointsCredit[frameIdx][jIdx] = jointsCredit[frameIdx - 1][jIdx] * 0.85;
			}
			else {
				jointsCredit[frameIdx][jIdx] = jointsCredit[frameIdx - 1][jIdx] + 1.5;
			}
			totalCredits[frameIdx] += jointsCredit[frameIdx][jIdx];
		}
	}

	std::cout << "Total Credits: " << totalCredits[frameIdx] << std::endl;
}