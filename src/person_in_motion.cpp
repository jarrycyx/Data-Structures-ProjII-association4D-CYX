#include "person_in_motion.h"
#include "skel.h"
#include <iostream>


const int Person3DMotion::ERR_THRES_REBUILD = 1;		//error大于此值认为完全不可用，需要重建
const double Person3DMotion::ERR_THRES_CALIBRT = 0.2;	//error大于此值认为需要校准，小于此值认为是正常点
const double Person3DMotion::PEOPLE_DISP_THRES = 5;		//credit大于此值认为可以显示
const double Person3DMotion::JOINT_REBUILD_THRES = 2;	//可以用于重建的结点可靠度阈值



Person3DMotion::Person3DMotion()
{
	const_skels_A = {4, 4, 1, 1, 5, 6, 11, 12, 1, 0, 0, 2, 3, 7, 8, 13, 14, 9, 10, 5, 6, 11, 12, 15, 16, 0, 2, 3, 7, 8, 13, 14, 18, 17 };
	const_skels_B = {9, 10, 5, 6, 11, 12, 15, 16, 0, 2, 3, 7, 8, 13, 14, 18, 17, 4, 4, 1, 1, 5, 6, 11, 12, 1, 0, 0, 2, 3, 7, 8, 13, 14, 9 };
	for (int i = 0; i < FRAME_SIZE; i++)
	{
		totalCredits[i] = 0;
		inViewFlag[i] = false;
		predictionFlag[i] = false;
		jointsAcceleration[i].resize(GetSkelDef().jointSize);
		jointsVelocity[i].resize(GetSkelDef().jointSize);
		jointsStatus[i].resize(GetSkelDef().jointSize);
		rebuildStatus[i].resize(GetSkelDef().jointSize);
		jointsCredit[i].resize(GetSkelDef().jointSize);
		jointsFineLocation[i].resize(GetSkelDef().jointSize);
		const_length.resize(const_skels_A.size());
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
		{
			jointsAcceleration[i][jIdx] = Eigen::VectorXd::Constant(3, 0);
			jointsVelocity[i][jIdx] = Eigen::VectorXd::Constant(3, 0);
			jointsFineLocation[i][jIdx] = Eigen::VectorXd::Constant(3, 0);
			jointsCredit[i][jIdx] = 0;
			jointsStatus[i][jIdx] = 0;
		}
	}
}


void Person3DMotion::dfsJoints(int frameIdx, int thisJoint)
{
	for (int cIdx = 0; cIdx < const_skels_A.size(); cIdx++)
	{
		if (const_skels_A[cIdx] == thisJoint)
		{
			if (rebuildStatus[frameIdx][const_skels_B[cIdx]] > 0)
			{
				//jointsFineLocation[frameIdx][const_skels_B[cIdx]] = jointsLocation[frameIdx][const_skels_B[cIdx]];
				if (jointsCredit[frameIdx - 1][const_skels_B[cIdx]] > JOINT_REBUILD_THRES && jointsFineLocation[frameIdx - 1][const_skels_A[cIdx]].norm() != 0 && jointsFineLocation[frameIdx - 1][const_skels_B[cIdx]].norm() != 0 && jointsFineLocation[frameIdx][const_skels_A[cIdx]].norm() != 0)
				{
					Eigen::VectorXd lastRelativeLoc = jointsFineLocation[frameIdx - 1][const_skels_B[cIdx]] - jointsFineLocation[frameIdx - 1][const_skels_A[cIdx]];
					jointsFineLocation[frameIdx][const_skels_B[cIdx]] = jointsFineLocation[frameIdx][const_skels_A[cIdx]] + lastRelativeLoc;
					rebuildStatus[frameIdx][const_skels_B[cIdx]] = 0;
					dfsJoints(frameIdx, const_skels_B[cIdx]);
				}
			}
		}
	}

}

void Person3DMotion::RebuildPersonSkels(int frameIdx)
{
	if (inViewFlag[frameIdx] == 0)
	{
		predictionFlag[frameIdx] = 1;
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
		{
			Eigen::VectorXd lastVec = jointsVelocity[frameIdx - 1][jIdx];
			Eigen::VectorXd lastAccel = jointsAcceleration[frameIdx - 1][jIdx];
			Eigen::VectorXd predLoc = jointsLocation[frameIdx - 1][jIdx] + lastVec + lastAccel;
			jointsFineLocation[frameIdx][jIdx] = predLoc;
			jointsCredit[frameIdx][jIdx] = jointsCredit[frameIdx - 1][jIdx] / 2;
			totalCredits[frameIdx] += jointsCredit[frameIdx][jIdx];
		}
	}
	else {
		bool hasEfficientJoint = false;
		int startJoint = 0;
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
			rebuildStatus[frameIdx][jIdx] = jointsStatus[frameIdx][jIdx];
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
		{
			if (jointsStatus[frameIdx][jIdx] == 0)
			{
				startJoint = jIdx;
				hasEfficientJoint = true;
				dfsJoints(frameIdx, startJoint);
			}
		}
		if (!hasEfficientJoint)
		{
			Eigen::VectorXd mainBodyVec = (jointsVelocity[frameIdx][0] + jointsVelocity[frameIdx][1] + jointsVelocity[frameIdx][2]) / 3;
			for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
			{
				jointsFineLocation[frameIdx][jIdx] = jointsFineLocation[frameIdx - 1][jIdx];
				if (jointsCredit[frameIdx][0] + jointsCredit[frameIdx][0] + jointsCredit[frameIdx][0] > 4)
					jointsFineLocation[frameIdx][jIdx] += mainBodyVec;
			}
		}
	}
}





//计算投影
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

//计算投影
Person2D Person3DMotion::ProjSkelFineLocation(int frameIdx, const Eigen::Matrix<float, 3, 4>& proj) const {
	Person2D person;
	Eigen::Matrix4Xf joints;
	joints = Eigen::Matrix4Xf::Zero(4, GetSkelDef().jointSize);
	for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
	{
		joints(0, jIdx) = jointsFineLocation[frameIdx][jIdx][0];
		joints(1, jIdx) = jointsFineLocation[frameIdx][jIdx][1];
		joints(2, jIdx) = jointsFineLocation[frameIdx][jIdx][2];
		joints(3, jIdx) = (jointsFineLocation[frameIdx][jIdx][0] != 0
			&& jointsFineLocation[frameIdx][jIdx][1] != 0
			&& jointsFineLocation[frameIdx][jIdx][2] != 0)
			&& jointsCredit[frameIdx][jIdx] >= 0.2;
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
		if (const_length[cIdx] == 0) 
			const_length[cIdx] = delta.norm();
		else const_length[cIdx] = const_length[cIdx] * 0.9 + delta.norm() * 0.1;
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
		jointsFineLocation[frameIdx][jIdx] = jointsLocation[frameIdx][jIdx];
		if (frameIdx != 0 && inViewFlag[frameIdx - 1])
		{
			Eigen::VectorXd r2 = jointsLocation[frameIdx][jIdx];
			Eigen::VectorXd r1 = jointsLocation[frameIdx - 1][jIdx];
			if (jointsLocation[frameIdx - 1][jIdx][0] != 0 && jointsLocation[frameIdx - 1][jIdx][1] != 0 && jointsLocation[frameIdx - 1][jIdx][2] != 0)
				if (jointsLocation[frameIdx][jIdx][0] != 0 && jointsLocation[frameIdx][jIdx][1] != 0 && jointsLocation[frameIdx][jIdx][2] != 0)
					if (jointsCredit[frameIdx - 1][jIdx] > 2 && (r2 - r1).norm() < 0.1)
						jointsVelocity[frameIdx][jIdx] = r2 - r1;
		}
		if (frameIdx > 1 && inViewFlag[frameIdx - 1])
		{
			Eigen::VectorXd v2 = jointsVelocity[frameIdx][jIdx];
			Eigen::VectorXd v1 = jointsVelocity[frameIdx - 1][jIdx];
			if (jointsVelocity[frameIdx - 1][jIdx][0] != 0 && jointsVelocity[frameIdx - 1][jIdx][0] != 0 && jointsVelocity[frameIdx - 1][jIdx][0] != 0)
				if (jointsVelocity[frameIdx][jIdx][0] != 0 && jointsVelocity[frameIdx][jIdx][1] != 0 && jointsVelocity[frameIdx][jIdx][2] != 0)
					if (jointsCredit[frameIdx - 1][jIdx] > 3 && (v2 - v1).norm() < 0.1)
						jointsAcceleration[frameIdx][jIdx] = v2 - v1;
		}
		
	}


	for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
	{
		if (frameIdx != 0 && inViewFlag[frameIdx - 1])
		{
			Eigen::VectorXd lastVec = jointsVelocity[frameIdx - 1][jIdx];
			Eigen::VectorXd lastAccel = jointsAcceleration[frameIdx - 1][jIdx];
			Eigen::VectorXd predLoc = jointsFineLocation[frameIdx - 1][jIdx] + lastVec + lastAccel;
			Eigen::VectorXd error = predLoc - jointsLocation[frameIdx][jIdx];
			if (jointsLocation[frameIdx - 1][jIdx][0] == 0)
				error = Eigen::VectorXd::Constant(3, 0.01);
			double jointProb = person.joints(3, jIdx);
			double totalError = abs(error[0]) + abs(error[1]) + abs(error[2]) + 0.03 / (0.01 + jointProb);
			if (totalError > ERR_THRES_REBUILD)
			{
				jointsCredit[frameIdx][jIdx] = jointsCredit[frameIdx - 1][jIdx] * 0.6;
				jointsFineLocation[frameIdx][jIdx] = predLoc;
				jointsStatus[frameIdx][jIdx] = 2;
			}
			else if (totalError > ERR_THRES_CALIBRT)
			{
				jointsCredit[frameIdx][jIdx] = jointsCredit[frameIdx - 1][jIdx] * 0.85;
				jointsFineLocation[frameIdx][jIdx] = predLoc * 0.5 + jointsLocation[frameIdx][jIdx] * (1 - 0.5);
				jointsStatus[frameIdx][jIdx] = 1;
			}
			else {
				jointsCredit[frameIdx][jIdx] = jointsCredit[frameIdx - 1][jIdx] + 1.5;
				jointsFineLocation[frameIdx][jIdx] = jointsLocation[frameIdx][jIdx];
				jointsStatus[frameIdx][jIdx] = 0;
				CalculateConstraintsSkels(frameIdx, person);
			}
			totalCredits[frameIdx] += jointsCredit[frameIdx][jIdx];
		}
	}

	RebuildPersonSkels(frameIdx);

	std::cout << "Total Credits: " << totalCredits[frameIdx] << std::endl;
}