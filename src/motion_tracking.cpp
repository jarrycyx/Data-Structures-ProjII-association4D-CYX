#include "motion_tracking.h"
#include "skel.h"




void MotionTracking::AddFrame(Associater* associater, int frameIdx)
{
	std::vector<Person3DMotion*> m_personsInFrame;
	int personNum = associater->GetPersons3D().size();
	persons.emplace_back(new Person3DMotion());
	persons[0]->CalculateNewFrameMotion(frameIdx, associater->GetPersons3D()[0]);
	currentFrame = frameIdx;
	/*
	for (int i = 0; i < personNum; i++)
	{
		Person3D person3D = associater->GetPersons3D()[i];
		int existing = false;
		for (int i = 0; i < GetSkelDef().jointSize; i++)//如果数据全部为0，说明这个人不存在
			existing |= (person3D.joints(0, i) || person3D.joints(1, i) || person3D.joints(2, i));
		
		if (existing)
		{
			m_personsInFrame.emplace_back(new Person3DMotion(person3D));
			//CalculateMotionStatus(frameIdx);
			currentFrame = frameIdx;
		}
	}*/
}

MotionTracking::MotionTracking(): currentFrame(0)
{

}