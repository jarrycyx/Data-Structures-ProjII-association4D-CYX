#include "association.h"
#include "color_util.h"
#include <opencv2/opencv.hpp>
#include <json/json.h>
#include <fstream>
#include <iostream>
#include "motion_tracking.h"

//test

std::map<std::string, Camera> ParseCameras(const std::string& filename)
{
	Json::Value json;
	std::ifstream fs(filename);
	if (!fs.is_open()) {
		std::cerr << "json file not exist: " << filename << std::endl;
		std::abort();
	}

	std::string errs;
	Json::parseFromStream(Json::CharReaderBuilder(), fs, &json, &errs);
	fs.close();

	if (errs != "") {
		std::cerr << "json read file error: " << errs << std::endl;
		std::abort();
	}

	std::map<std::string, Camera> cameras;
	for (auto camIter = json.begin(); camIter != json.end(); camIter++) {
		Camera camera;

		for (int i = 0; i < 9; i++)
			camera.K(i / 3, i % 3) = (*camIter)["K"][i].asFloat();

		Eigen::Vector3f r;
		for (int i = 0; i < 3; i++)
			r(i) = (*camIter)["R"][i].asFloat();
		camera.R = Eigen::AngleAxisf(r.norm(), r.normalized()).matrix();

		for (int i = 0; i < 3; i++)
			camera.T(i) = (*camIter)["T"][i].asFloat();

		camera.Update();
		cameras.insert(std::make_pair(camIter.key().asString(), camera));
	}
	return cameras;
}


std::vector<std::vector<SkelDetection>> ParseDetections(const std::string& filename)
{

	std::ofstream csvOut;
	csvOut.open("../debug/FramesData.csv", std::ios::out);
	csvOut << "";
	csvOut.close();

	std::ifstream fs(filename);
	if (!fs.is_open()) {
		std::cerr << "json file not exist: " << filename << std::endl;
		std::abort();
	}

	int frameSize, camSize;
	fs >> frameSize >> camSize;

	std::vector<std::vector<SkelDetection>> detections(frameSize, std::vector<SkelDetection>(camSize));
	for (int frameIdx = 0; frameIdx < frameSize; frameIdx++) {
		for (int camIdx = 0; camIdx < camSize; camIdx++) {
			SkelDetection& detection = detections[frameIdx][camIdx];
			for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
				int jSize;
				fs >> jSize;
				detection.joints[jIdx].resize(3, jSize);
				for (int i = 0; i < 3; i++)
					for (int j = 0; j < jSize; j++)
						fs >> detection.joints[jIdx](i, j);
			}
			for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {//一个人体模型中的某条边
				const int jAIdx = GetSkelDef().pafDict(0, pafIdx);//这条边的一个顶点
				const int jBIdx = GetSkelDef().pafDict(1, pafIdx);//另一个顶点
				detection.pafs[pafIdx].resize(detection.joints[jAIdx].cols(), detection.joints[jBIdx].cols());
				//std::cout << detection.joints[jAIdx].cols() << " " << detection.joints[jBIdx].cols() << std::endl;
				//该图中检测出很多个该类边连接的顶点，比如2+2，3+3个
				//取得连接概率关系2x2，3x3，但n个人只会有n条边
				for (int i = 0; i < detection.pafs[pafIdx].rows(); i++)
					for (int j = 0; j < detection.pafs[pafIdx].cols(); j++)
						fs >> detection.pafs[pafIdx](i, j);
			}
		}
	}
	fs.close();
	return detections;
}



void SaveResult(const int& frameIdx, const std::vector<cv::Mat>& images, const std::vector<SkelDetection>& detections, const std::vector<Camera>& cameras,
	const std::vector<std::vector<Person2D>>& persons2D, const std::vector<Person3D>& persons3D, const MotionTracking& tracking)
{
	const int rows = 2;
	const int cols = (cameras.size() + rows - 1) / rows;
	const Eigen::Vector2i imgSize(images.begin()->cols, images.begin()->rows);
	const int jointRadius = round(imgSize.x() * 1.f / 128.f);
	const int pafThickness = round(imgSize.x() * 1.f / 256.f);
	const float textScale = sqrtf(imgSize.x() / 1024.f);
	cv::Mat detectImg(rows * imgSize.y(), cols * imgSize.x(), CV_8UC3);
	cv::Mat assocImg(rows * imgSize.y(), cols * imgSize.x(), CV_8UC3);
	cv::Mat reprojImg(rows * imgSize.y(), cols * imgSize.x(), CV_8UC3);
	cv::Mat trackImg(rows * imgSize.y(), cols * imgSize.x(), CV_8UC3);
	cv::Mat predImg(rows * imgSize.y(), cols * imgSize.x(), CV_8UC3);
	
	for (int camIdx = 0; camIdx < cameras.size(); camIdx++) {
		cv::Rect roi(camIdx % cols * imgSize.x(), camIdx / cols * imgSize.y(), imgSize.x(), imgSize.y());
		images[camIdx].copyTo(detectImg(roi));
		images[camIdx].copyTo(assocImg(roi));
		images[camIdx].copyTo(reprojImg(roi));
		images[camIdx].copyTo(trackImg(roi));
		images[camIdx].copyTo(predImg(roi));
		/*
		// draw detection
		const SkelDetection& detection = detections[camIdx];
		for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {
			const int jaIdx = GetSkelDef().pafDict(0, pafIdx);
			const int jbIdx = GetSkelDef().pafDict(1, pafIdx);
			for (int jaCandiIdx = 0; jaCandiIdx < detection.joints[jaIdx].cols(); jaCandiIdx++) {
				for (int jbCandiIdx = 0; jbCandiIdx < detection.joints[jbIdx].cols(); jbCandiIdx++) {
					if (detection.joints[jaIdx](2, jaCandiIdx) > 0.f && detection.joints[jbIdx](2, jbCandiIdx) > 0.f) {
						const int thickness = round(detection.pafs[pafIdx](jaCandiIdx, jbCandiIdx) * pafThickness);
						//连接概率不为0则画出的线可见
						if (thickness > 0) {
							const cv::Point jaPos(round(detection.joints[jaIdx](0, jaCandiIdx) * imgSize.x() - 0.5f), round(detection.joints[jaIdx](1, jaCandiIdx) * imgSize.y() - 0.5f));
							const cv::Point jbPos(round(detection.joints[jbIdx](0, jbCandiIdx) * imgSize.x() - 0.5f), round(detection.joints[jbIdx](1, jbCandiIdx) * imgSize.y() - 0.5f));
							cv::line(detectImg(roi), jaPos, jbPos, ColorUtil::GetColor("gray"), thickness);
						}
					}
				}
			}
		}

		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
			for (int candiIdx = 0; candiIdx < detection.joints[jIdx].cols(); candiIdx++) {
				const cv::Point jPos(round(detection.joints[jIdx](0, candiIdx) * imgSize.x() - 0.5f), round(detection.joints[jIdx](1, candiIdx) * imgSize.y() - 0.5f));
				const int radius = round(detection.joints[jIdx](2, candiIdx) * jointRadius);
				if (radius > 0) {
					cv::circle(detectImg(roi), jPos, radius, ColorUtil::GetColor("white"), 2);
					cv::putText(detectImg(roi), std::to_string(jIdx), jPos, cv::FONT_ITALIC, textScale, ColorUtil::GetColor("white"));
				}
			}
		}

		// draw assoc
		for (int pIdx = 0; pIdx < persons2D[camIdx].size(); pIdx++) {
			const Person2D& person = persons2D[camIdx][pIdx];
			const cv::Scalar& color = ColorUtil::GetColor(pIdx);
			for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
				if (person.joints(2, jIdx) < FLT_EPSILON)
					continue;
				cv::Point jPos(round(person.joints(0, jIdx) * imgSize.x() - 0.5f), round(person.joints(1, jIdx) * imgSize.y() - 0.5f));
				cv::circle(assocImg(roi), jPos, jointRadius, color, -1);
				cv::putText(assocImg(roi), std::to_string(jIdx), jPos, cv::FONT_ITALIC, textScale, ColorUtil::GetColor("white"));
			}

			for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {
				const int jaIdx = GetSkelDef().pafDict(0, pafIdx);
				const int jbIdx = GetSkelDef().pafDict(1, pafIdx);
				if (person.joints(2, jaIdx) > FLT_EPSILON&& person.joints(2, jbIdx) > FLT_EPSILON) {
					const cv::Point jaPos(round(person.joints(0, jaIdx) * imgSize.x() - 0.5f), round(person.joints(1, jaIdx) * imgSize.y() - 0.5f));
					const cv::Point jbPos(round(person.joints(0, jbIdx) * imgSize.x() - 0.5f), round(person.joints(1, jbIdx) * imgSize.y() - 0.5f));
					cv::line(assocImg(roi), jaPos, jbPos, color, pafThickness);
				}
			}
		}

		// draw proj
		for (int pIdx = 0; pIdx < persons3D.size(); pIdx++) {
			const Person2D& person = persons3D[pIdx].ProjSkel(cameras[camIdx].proj);
			const cv::Scalar& color = ColorUtil::GetColor(pIdx);
			for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
				if (person.joints(2, jIdx) < FLT_EPSILON)
					continue;
				cv::Point jPos(round(person.joints(0, jIdx) * imgSize.x() - 0.5f), round(person.joints(1, jIdx) * imgSize.y() - 0.5f));
				cv::circle(reprojImg(roi), jPos, jointRadius, color, -1);
				cv::putText(reprojImg(roi), std::to_string(jIdx), jPos, cv::FONT_ITALIC, textScale, ColorUtil::GetColor("white"));
			}

			for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {
				const int jaIdx = GetSkelDef().pafDict(0, pafIdx);
				const int jbIdx = GetSkelDef().pafDict(1, pafIdx);
				if (person.joints(2, jaIdx) > FLT_EPSILON&& person.joints(2, jbIdx) > FLT_EPSILON) {
					const cv::Point jaPos(round(person.joints(0, jaIdx) * imgSize.x() - 0.5f), round(person.joints(1, jaIdx) * imgSize.y() - 0.5f));
					const cv::Point jbPos(round(person.joints(0, jbIdx) * imgSize.x() - 0.5f), round(person.joints(1, jbIdx) * imgSize.y() - 0.5f));
					cv::line(reprojImg(roi), jaPos, jbPos, color, pafThickness);
				}
			}
		}
		
		
		// draw track
		for (int pIdx = 0; pIdx < tracking.persons.size(); pIdx++) 
			if (tracking.persons[pIdx]->inViewFlag[frameIdx]){
				const Person2D& person = tracking.persons[pIdx]->ProjSkel(frameIdx, cameras[camIdx].proj);
				const cv::Scalar& color = ColorUtil::GetColor(pIdx);
				for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
					if (person.joints(2, jIdx) < FLT_EPSILON)
						continue;
					cv::Point jPos(round(person.joints(0, jIdx) * imgSize.x() - 0.5f), round(person.joints(1, jIdx) * imgSize.y() - 0.5f));
					cv::circle(trackImg(roi), jPos, jointRadius, color, -1);
					cv::putText(trackImg(roi), std::to_string(jIdx), jPos, cv::FONT_ITALIC, textScale, ColorUtil::GetColor("white"));
				}

				for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {
					const int jaIdx = GetSkelDef().pafDict(0, pafIdx);
					const int jbIdx = GetSkelDef().pafDict(1, pafIdx);
					if (person.joints(2, jaIdx) > FLT_EPSILON&& person.joints(2, jbIdx) > FLT_EPSILON) {
						const cv::Point jaPos(round(person.joints(0, jaIdx) * imgSize.x() - 0.5f), round(person.joints(1, jaIdx) * imgSize.y() - 0.5f));
						const cv::Point jbPos(round(person.joints(0, jbIdx) * imgSize.x() - 0.5f), round(person.joints(1, jbIdx) * imgSize.y() - 0.5f));
						cv::line(trackImg(roi), jaPos, jbPos, color, pafThickness);
					}
				}
			}
			*/

		for (int pIdx = 0; pIdx < tracking.persons.size(); pIdx++)
			if ((tracking.persons[pIdx]->inViewFlag[frameIdx] || tracking.persons[pIdx]->predictionFlag[frameIdx])
				&& tracking.persons[pIdx]->totalCredits[frameIdx] > Person3DMotion::PEOPLE_DISP_THRES) {
				const Person2D& person = tracking.persons[pIdx]->ProjSkelFineLocation(frameIdx, cameras[camIdx].proj);
				const cv::Scalar& color = ColorUtil::GetColor(pIdx);
				for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
					if (person.joints(2, jIdx) < FLT_EPSILON)
						continue;
					cv::Point jPos(round(person.joints(0, jIdx) * imgSize.x() - 0.5f), round(person.joints(1, jIdx) * imgSize.y() - 0.5f));
					cv::circle(predImg(roi), jPos, jointRadius, color, -1);
					cv::putText(predImg(roi), std::to_string(jIdx), jPos, cv::FONT_ITALIC, textScale, ColorUtil::GetColor(tracking.persons[pIdx]->jointsStatus[frameIdx][jIdx]));
				}

				for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {
					const int jaIdx = GetSkelDef().pafDict(0, pafIdx);
					const int jbIdx = GetSkelDef().pafDict(1, pafIdx);
					if (person.joints(2, jaIdx) > FLT_EPSILON&& person.joints(2, jbIdx) > FLT_EPSILON) {
						const cv::Point jaPos(round(person.joints(0, jaIdx) * imgSize.x() - 0.5f), round(person.joints(1, jaIdx) * imgSize.y() - 0.5f));
						const cv::Point jbPos(round(person.joints(0, jbIdx) * imgSize.x() - 0.5f), round(person.joints(1, jbIdx) * imgSize.y() - 0.5f));
						cv::line(predImg(roi), jaPos, jbPos, color, pafThickness);
					}
				}
			}
	}
	//cv::imwrite("D:/11PRojects/Data.Structures.ProjII.association4D/OutputImages/detect/dt_" + std::to_string(frameIdx) + ".jpg", detectImg);
	//cv::imwrite("D:/11PRojects/Data.Structures.ProjII.association4D/OutputImages/assoc/as_" + std::to_string(frameIdx) + ".jpg", assocImg);
	//cv::imwrite("D:/11PRojects/Data.Structures.ProjII.association4D/OutputImages/reproj/rp_" + std::to_string(frameIdx) + ".jpg", reprojImg);
	//cv::imwrite("D:/11PRojects/Data.Structures.ProjII.association4D/OutputImages/track/tk_" + std::to_string(frameIdx) + ".jpg", trackImg);
	cv::imwrite("D:/11PRojects/Data.Structures.ProjII.association4D/OutputImages/pred/pd_" + std::to_string(frameIdx) + ".jpg", predImg);
}

void SaveAssociationCsv(Associater* associater, int frameIndex)
{
	std::ofstream csvOut;
	csvOut.open("../debug/FramesData.csv", std::ios::app);
	csvOut << "Frame " << frameIndex << std::endl;

	for (int i = 0; i < GetSkelDef().jointSize; i++)
		csvOut << i << "x," << i << "y," << i << "z,";
	csvOut << std::endl;
	for (int i = 0; i < associater->GetPersons3D().size(); i++)
	{
		Person3D thisPerson = associater->GetPersons3D()[i];
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
		{
			csvOut << thisPerson.joints(0, jIdx) << ",";
			csvOut << thisPerson.joints(1, jIdx) << ",";
			csvOut << thisPerson.joints(2, jIdx) << ",";
		}
		csvOut << std::endl;
	}

}

void SaveMotionCsv(MotionTracking* motionTracking)
{
	std::ofstream csvOut;
	csvOut.open("../debug/MotionData.csv", std::ios::out);

	for (int frameIdx = 0; frameIdx < motionTracking->currentFrame; frameIdx++)
	{
		std::ofstream frameCsv;
		frameCsv.open(("../debug/FrameMotion/" + std::to_string(frameIdx) + ".csv"), std::ios::out);

		csvOut << "Frame " << frameIdx << std::endl;
		
		csvOut << "no, " << "TTLcre,";
		for (int i = 0; i < GetSkelDef().jointSize; i++)
			csvOut << i << "cre," << i << "x," << i << "y," << i << "z," << i << "fx," << i << "fy," << i << "fz," << i << "vx," << i << "vy," << i << "vz," << i << "ax," << i << "ay," << i << "az,";
		csvOut << std::endl;
		for (int p = 0; p < motionTracking->persons.size(); p++)
		{
			Person3DMotion* thisPerson = motionTracking->persons[p];
			if (thisPerson->inViewFlag[frameIdx] || thisPerson->predictionFlag[frameIdx]) {
				csvOut << p << ", ";
				csvOut << thisPerson->totalCredits[frameIdx] << ", ";
				for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
				{
					csvOut << thisPerson->jointsCredit[frameIdx][jIdx] << ",";
					csvOut << thisPerson->jointsLocation[frameIdx][jIdx](0) << ",";
					csvOut << thisPerson->jointsLocation[frameIdx][jIdx](1) << ",";
					csvOut << thisPerson->jointsLocation[frameIdx][jIdx](2) << ",";
					csvOut << thisPerson->jointsFineLocation[frameIdx][jIdx](0) << ",";
					csvOut << thisPerson->jointsFineLocation[frameIdx][jIdx](1) << ",";
					csvOut << thisPerson->jointsFineLocation[frameIdx][jIdx](2) << ",";
					csvOut << thisPerson->jointsVelocity[frameIdx][jIdx](0) << ",";
					csvOut << thisPerson->jointsVelocity[frameIdx][jIdx](1) << ",";
					csvOut << thisPerson->jointsVelocity[frameIdx][jIdx](2) << ",";
					csvOut << thisPerson->jointsAcceleration[frameIdx][jIdx](0) << ",";
					csvOut << thisPerson->jointsAcceleration[frameIdx][jIdx](1) << ",";
					csvOut << thisPerson->jointsAcceleration[frameIdx][jIdx](2);
					if (jIdx != GetSkelDef().jointSize - 1) csvOut << ",";

					frameCsv << thisPerson->jointsLocation[frameIdx][jIdx](0) << ",";
					frameCsv << thisPerson->jointsLocation[frameIdx][jIdx](1) << ",";
					frameCsv << thisPerson->jointsLocation[frameIdx][jIdx](2);
					if (jIdx != GetSkelDef().jointSize - 1) frameCsv << ",";
				}
				csvOut << std::endl;
				frameCsv << std::endl;
			}
		}
		frameCsv.close();
	}
	csvOut.close();

}


int main()
{
	// load data and detections.
	std::map<std::string, Camera> _cameras = ParseCameras("../data/calibration.json");
	std::vector<Camera> cameras;
	std::vector<cv::Mat> rawImgs;
	std::vector<cv::VideoCapture> videos;
	for (const auto& iter : _cameras) {
		cameras.emplace_back(iter.second);
		videos.emplace_back(cv::VideoCapture("../data/" + iter.first + ".mpeg"));
		rawImgs.emplace_back(cv::Mat());
	}//52ms

	std::vector<std::vector<SkelDetection>> detections = ParseDetections("../data/detection.txt");

	// init
	Associater associater(cameras);
	MotionTracking motionTracking;

	for (int frameIdx = 0; frameIdx < detections.size(); frameIdx++) {
		//for (int frameIdx = 0; frameIdx < 50; frameIdx++) {
		for (int camIdx = 0; camIdx < cameras.size(); camIdx++)
			videos[camIdx] >> rawImgs[camIdx];

		associater.SetDetection(detections[frameIdx]);//设置数据的格式，1ms
		associater.ConstructJointRays();//根据2D数据，计算当前帧，每台相机成像的3D坐标投影方向，2ms
		associater.ConstructJointEpiEdges();//计算当前帧，相邻两台相机的投影线间距，3ms
		associater.ClusterPersons2D();//推断2D图形中属于同一个人的点，2ms
		associater.ProposalCollocation();//算出所有可能的分配，注意是所有3D人、所有视图、视图上的所有2D人的可能分配，1ms
		associater.ClusterPersons3D();//暴力推断多视图的每个点分别都是哪个人的，7ms
		associater.ConstructPersons();//完成对每个人的每个点的标记，11ms
		//
		std::cout << "Frame Idx:" << std::to_string(frameIdx) << std::endl;
		motionTracking.AddFrame(&associater, frameIdx);

		SaveAssociationCsv(&associater, frameIdx);
		SaveResult(frameIdx, rawImgs, detections[frameIdx], cameras, associater.GetPersons2D(), associater.GetPersons3D(), motionTracking);

		std::cout << std::endl;
	}

	SaveMotionCsv(&motionTracking);

	int a;
	std::cin >> a;
	return 0;
}


/*
ffmpeg -i D:\11PRojects\Data.Structures.ProjII.association4D\OutputImages\track\tk_%d.jpg -vcodec mpeg4 D:\11PRojects\Data.Structures.ProjII.association4D\OutputImages\track.mp4 -r 25
ffmpeg -i D:\11PRojects\Data.Structures.ProjII.association4D\OutputImages\pred\pd_%d.jpg -vcodec mpeg4 D:\11PRojects\Data.Structures.ProjII.association4D\OutputImages\pred.mp4 -r 25
*/