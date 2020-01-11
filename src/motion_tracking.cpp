#include "motion_tracking.h"
#include "skel.h"
#include <iostream>
#include "kuhn_munkras_algo.h"

double MotionTracking::CalcFrameMotionLoss(int personIdx, int thisFramePersonIdx)
{
	double loss = 0;
	int jointsNum = 0;
	for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
	{
		if (persons[personIdx]->jointsLocation[currentFrame - 1][jIdx].norm() > 0) {
			jointsNum++;
			Eigen::VectorXd predV;
			predV.resize(3);
			predV = persons[personIdx]->jointsVelocity[currentFrame - 1][jIdx] + persons[personIdx]->jointsAcceleration[currentFrame - 1][jIdx];
			Eigen::VectorXd predX;
			predX.resize(3);
			predX = persons[personIdx]->jointsFineLocation[currentFrame - 1][jIdx];
			loss += abs(predX[0] - thisFramePersons[thisFramePersonIdx].joints(0, jIdx));
			loss += abs(predX[1] - thisFramePersons[thisFramePersonIdx].joints(1, jIdx));
			loss += abs(predX[2] - thisFramePersons[thisFramePersonIdx].joints(2, jIdx));
		}
	}

	return (loss / (jointsNum * jointsNum)) * GetSkelDef().jointSize * GetSkelDef().jointSize;  //对点的数量设置惩罚
}


void MotionTracking::MyKuhnMunkrasAlgo(double (&losses)[MAX_PERSON][MAX_PERSON], int* corres, int* avail)
{
	int thisFramePersonNum = thisFramePersons.size();
	int totalMotionPersonNum = persons.size();
	KuhnMunkrasAlgo kmAlgo;
	kmAlgo.loadData(totalMotionPersonNum, thisFramePersonNum, losses, 0.01, 10.0);
	kmAlgo.solve(avail, corres);
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

	int thisFrameAvailable[MAX_PERSON];
	int corresIdx[MAX_PERSON];
	for (int i = 0; i < MAX_PERSON; i++)
	{
		thisFrameAvailable[i] = -1;
		corresIdx[i] = -1;
	}

	for (int i = 0; i < totalMotionPersonNum; i++)
	{
		for (int j = 0; j < thisFramePersons.size(); j++)
		{
			if (persons[i]->inViewFlag[frameIdx - 1])
				losses[i][j] = CalcFrameMotionLoss(i, j);
			else
				losses[i][j] = MAX;
		}
	}
	
	MyKuhnMunkrasAlgo(losses, corresIdx, thisFrameAvailable);

	for (int i = 0; i < thisFramePersonNum; i++)
	{
		if (thisFrameAvailable[i] == -1) { //还没有被分配的

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
		else
			if (thisFrameAvailable[i] != -1)
				persons[thisFrameAvailable[i]]->CalculateNewFrameMotion(frameIdx, thisFramePersons[i]);
	}

	int actualPersonNum = 0;

	for (int i = 0; i < totalMotionPersonNum; i++)
	{
		if (persons[i]->inViewFlag[frameIdx])
			actualPersonNum += 1;
		else if (frameIdx && persons[i]->inViewFlag[frameIdx - 1] && persons[i]->totalCredits[frameIdx - 1] > 20)
			persons[i]->RebuildPersonSkels(frameIdx);

	}
	std::cout << "Person Num: " << actualPersonNum << std::endl;

}

MotionTracking::MotionTracking() : currentFrame(0)
{

}