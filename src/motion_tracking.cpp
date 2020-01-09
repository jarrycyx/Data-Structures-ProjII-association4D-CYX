#include "motion_tracking.h"
#include "skel.h"
#include <iostream>


double MotionTracking::CalcFrameMotionLoss(int personIdx, int thisFramePersonIdx)
{
	double loss = 0;
	for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
	{
		Eigen::VectorXd predV;
		predV.resize(3);
		predV = persons[personIdx]->jointsVelocity[currentFrame - 1][jIdx] + persons[personIdx]->jointsAcceleration[currentFrame - 1][jIdx];
		Eigen::VectorXd predX;
		predX.resize(3);
		predX = predV + persons[personIdx]->jointsLocation[currentFrame - 1][jIdx];
		loss += abs(predX[0] - thisFramePersons[thisFramePersonIdx].joints(0, jIdx));
		loss += abs(predX[1] - thisFramePersons[thisFramePersonIdx].joints(1, jIdx));
		loss += abs(predX[2] - thisFramePersons[thisFramePersonIdx].joints(2, jIdx));
	}
	return loss;
}


void MotionTracking::AddFrame(Associater* associater, int frameIdx)
{
	int personNum = associater->GetPersons3D().size();
	thisFramePersons.clear();
	for (int i = 0; i < personNum; i++)
	{
		Person3D person3D = associater->GetPersons3D()[i];
		int existing = false;
		for (int i = 0; i < GetSkelDef().jointSize; i++)//如果数据全部为0，说明这个人不存在
			existing |= (person3D.joints(0, i) || person3D.joints(1, i) || person3D.joints(2, i));

		if (existing)
		{
			thisFramePersons.emplace_back(person3D);
			currentFrame = frameIdx;
		}
	}

	double losses[MAX_PERSON][MAX_PERSON];
	int thisFramePersonNum = thisFramePersons.size();
	int totalMotionPersonNum = persons.size();
	bool thisFrameAvailable[MAX_PERSON];
	int corresIdx[MAX_PERSON];
	for (int i = 0; i < MAX_PERSON; i++)
	{
		thisFrameAvailable[i] = 1;
		corresIdx[i] = -1;
	}

	for (int i = 0; i < totalMotionPersonNum; i++)
	{
		if (persons[i]->inViewFlag[frameIdx - 1])
		for (int j = 0; j < thisFramePersons.size(); j++)
				losses[i][j] = CalcFrameMotionLoss(i, j);
	}
	
	//TO-DO：替换为匈牙利算法
	for (int i = 0; i < totalMotionPersonNum; i++)
	{
		int min = MAX;
		if (persons[i]->inViewFlag[frameIdx - 1])
			for (int j = 0; j < thisFramePersonNum; j++)
			{
				if (min > losses[i][j])
				{
					min = losses[i][j];
					corresIdx[i] = j;
				}
			}
		if (min < 5)
		{
			persons[i]->CalculateNewFrameMotion(frameIdx, thisFramePersons[corresIdx[i]]);
			thisFrameAvailable[corresIdx[i]] = 0;
		}
	}

	for (int i = 0; i < thisFramePersonNum; i++)
	{
		if (thisFrameAvailable[i]) {

			bool hasEmptySpot = false;
			int spot = 0;
			for (int j = 0; j < totalMotionPersonNum; j++)
				if (!persons[j]->inViewFlag[frameIdx])
				{
					hasEmptySpot = true;
					spot = j;
					break;
				}
			if (hasEmptySpot)
			{
				persons[spot]->CalculateNewFrameMotion(frameIdx, thisFramePersons[i]);
			}
			else
			{
				Person3DMotion* newMotionPerson = new Person3DMotion();
				newMotionPerson->CalculateNewFrameMotion(frameIdx, thisFramePersons[i]);
				persons.emplace_back(newMotionPerson);
			}
		}
	}

	int actualPersonNum = 0;

	for (int i = 0; i < totalMotionPersonNum; i++)
		actualPersonNum += (persons[i]->inViewFlag[frameIdx]);
	std::cout << actualPersonNum;

}

MotionTracking::MotionTracking() : currentFrame(0)
{

}