#pragma once

#include "skel.h"
#include "association.h"
#include "person_in_motion.h"

#define FRAME_SIZE 500
#define MAX_PERSON 100
#define MAX  0x3f3f3f

class MotionTracking
{
public:
	MotionTracking();
	void AddFrame(Associater* associater, int frameIdx);
	double CalcFrameMotionLoss(int personIdx, int thisFramePersonIdx);
	int currentFrame;
	std::vector<Person3DMotion*> persons;
	std::vector<Person3D> thisFramePersons;
};

