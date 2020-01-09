#pragma once

#include "skel.h"
#include "association.h"
#include "person_in_motion.h"

#define FRAME_SIZE 500

class MotionTracking
{
public:
	MotionTracking();
	void AddFrame(Associater* associater, int frameIdx);
	int currentFrame;
	std::vector<Person3DMotion*> persons;
};

