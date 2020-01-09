#include "association.h"
#include "math_util.h"
#include "color_util.h"
#include <sstream>
#include <numeric>
#include <algorithm>
#include <Eigen\Eigen>
#include <opencv2/opencv.hpp>


Associater::Associater(const std::vector<Camera>& _cams)
{
	m_cams = _cams;

	m_epiThresh = 0.2f;
	m_wEpi = 1.f;
	m_wView = 4.f;
	m_wPaf = 1.f;
	m_cPlaneTheta = 2e-3f;
	m_cViewCnt = 2.f;
	m_triangulateThresh = 0.05f;

	m_persons2D.resize(m_cams.size());
	m_personsMapByView.resize(m_cams.size());
	m_assignMap.resize(m_cams.size(), std::vector<Eigen::VectorXi>(GetSkelDef().jointSize));
	m_jointRays.resize(m_cams.size(), std::vector<Eigen::Matrix3Xf>(GetSkelDef().jointSize));
	m_jointEpiEdges.resize(GetSkelDef().jointSize);
	for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) 
		m_jointEpiEdges[jIdx].resize(m_cams.size(), std::vector<Eigen::MatrixXf>(m_cams.size()));
}


void Associater::SetDetection(const std::vector<SkelDetection>& detections)
{
	assert(m_cams.size() == detections.size());
	m_detections = detections;

	// reset assign map
	for (int view = 0; view < m_cams.size(); view++)
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
			m_assignMap[view][jIdx].setConstant(m_detections[view].joints[jIdx].cols(), -1);
			//-1表示未填入投影2D坐标


	for (int view = 0; view < m_cams.size(); view++) {
		m_persons2D[view].clear();
		m_personsMapByView[view].clear();
	}
	m_persons3D.clear();

}


void Associater::ConstructJointRays()
{
	for (int view = 0; view < m_cams.size(); view++) {
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {//遍历骨骼框架的每一类点
			const Eigen::Matrix3Xf& joints = m_detections[view].joints[jIdx];
			m_jointRays[view][jIdx].resize(3, joints.cols());//m_jointRays存储每个二维点解出的三维坐标
			for (int jCandiIdx = 0; jCandiIdx < joints.cols(); jCandiIdx++)//遍历一帧中，找出的若干个某一类点
				m_jointRays[view][jIdx].col(jCandiIdx) = m_cams[view].CalcRay(joints.block<2, 1>(0, jCandiIdx));
				//计算一台相机，一类点中每个点的3D投影方向
				//3D中的实际点可能在光心向投影方向连线上的任意位置
		}
	}
}


void Associater::ConstructJointEpiEdges()
{
	for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
		for (int viewA = 0; viewA < m_cams.size() - 1; viewA++) {
			for (int viewB = viewA + 1; viewB < m_cams.size(); viewB++) {//遍历两个相邻的相机
				Eigen::MatrixXf& epi = m_jointEpiEdges[jIdx][viewA][viewB];//从A和B两台相机中恢复的单值3D坐标
				const Eigen::Matrix3Xf& jointsA = m_detections[viewA].joints[jIdx];
				const Eigen::Matrix3Xf& jointsB = m_detections[viewB].joints[jIdx];//原始点数据
				const Eigen::Matrix3Xf& raysA = m_jointRays[viewA][jIdx];
				const Eigen::Matrix3Xf& raysB = m_jointRays[viewB][jIdx];//多值（一条投影直线）的解析数据
				epi.setConstant(jointsA.cols(), jointsB.cols(), -1.f);//行数为A相机中检测到某一类点的数目，列数为B相机中检测到某一类点的数目
				for (int jaCandiIdx = 0; jaCandiIdx < epi.rows(); jaCandiIdx++) {
					for (int jbCandiIdx = 0; jbCandiIdx < epi.cols(); jbCandiIdx++) {//两台相机，某一类点的所有点都要遍历
						const float dist = MathUtil::Line2LineDist(//两线间距，每条线由相机光心和3D预测投影方向组成
							m_cams[viewA].pos, raysA.col(jaCandiIdx), m_cams[viewB].pos, raysB.col(jbCandiIdx));
						if (dist < m_epiThresh)
							epi(jaCandiIdx, jbCandiIdx) = dist / m_epiThresh;
					}
				}
				m_jointEpiEdges[jIdx][viewB][viewA] = epi.transpose();//存储某一类点，所有相邻相机，所有点投影线的距离
			}
		}
	}
}



void Associater::ClusterPersons2D(const SkelDetection& detection, std::vector<Person2D>& persons, std::vector<Eigen::VectorXi>& assignMap)
{
	persons.clear();

	// generate valid pafs
	std::vector<std::tuple<float, int, int, int>> pafSet;
	for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {//人体骨骼模型的每一条连接边
		const int jaIdx = GetSkelDef().pafDict(0, pafIdx);//这条边的两个顶点编号
		const int jbIdx = GetSkelDef().pafDict(1, pafIdx);
		for (int jaCandiIdx = 0; jaCandiIdx < detection.joints[jaIdx].cols(); jaCandiIdx++) {//探测到了许多个这样的顶点
			for (int jbCandiIdx = 0; jbCandiIdx < detection.joints[jbIdx].cols(); jbCandiIdx++) {
				const float jaScore = detection.joints[jaIdx](2, jaCandiIdx);//该顶点对应的概率
				const float jbScore = detection.joints[jbIdx](2, jbCandiIdx);
				const float pafScore = detection.pafs[pafIdx](jaCandiIdx, jbCandiIdx);
				if (jaScore > 0.f && jbScore > 0.f && pafScore > 0.f)
					pafSet.emplace_back(std::make_tuple(pafScore, pafIdx, jaCandiIdx, jbCandiIdx));
					//两类点中，某两个点相连关系
			}
		}
	}
	std::sort(pafSet.rbegin(), pafSet.rend());//所有边排序

	// construct bodies use minimal spanning tree
	assignMap.resize(GetSkelDef().jointSize);
	for (int jIdx = 0; jIdx < assignMap.size(); jIdx++)
		assignMap[jIdx].setConstant(detection.joints[jIdx].cols(), -1);

	for (const auto& paf : pafSet) {
		const float pafScore = std::get<0>(paf);
		const int pafIdx = std::get<1>(paf);//骨骼框架中边的编号
		const int jaCandiIdx = std::get<2>(paf);//某类点中，某个点的序号
		const int jbCandiIdx = std::get<3>(paf);
		const int jaIdx = GetSkelDef().pafDict(0, pafIdx);//某类点在骨骼框架中的编号
		const int jbIdx = GetSkelDef().pafDict(1, pafIdx);

		int& aAssign = assignMap[jaIdx][jaCandiIdx];//某类点的某个点分配给某个人
		int& bAssign = assignMap[jbIdx][jbCandiIdx];

		// 1. A & B not assigned yet: Create new person
		if (aAssign == -1 && bAssign == -1) {
			Person2D person;
			person.joints.col(jaIdx) = detection.joints[jaIdx].col(jaCandiIdx);
			person.joints.col(jbIdx) = detection.joints[jbIdx].col(jbCandiIdx);
			person.pafs(pafIdx) = pafScore;
			aAssign = bAssign = persons.size();
			persons.emplace_back(person);
		}

		// 2. A assigned but not B: Add B to person with A (if no another B there) 
		// 3. B assigned but not A: Add A to person with B (if no another A there)
		else if ((aAssign >= 0 && bAssign == -1) || (aAssign == -1 && bAssign >= 0)) {
			const int assigned = aAssign >= 0 ? aAssign : bAssign;
			int& unassigned = aAssign >= 0 ? bAssign : aAssign;
			const int unassignedIdx = aAssign >= 0 ? jbIdx : jaIdx;
			const int unassignedCandiIdx = aAssign >= 0 ? jbCandiIdx : jaCandiIdx;

			Person2D& person = persons[assigned];
			if (person.joints(2, unassignedIdx) < FLT_EPSILON) {
				person.joints.col(unassignedIdx) = detection.joints[unassignedIdx].col(unassignedCandiIdx);
				person.pafs(pafIdx) = pafScore;
				unassigned = assigned;
			}
		}

		// 4. A & B already assigned to same person (circular/redundant PAF)
		else if (aAssign == bAssign)
			persons[aAssign].pafs(pafIdx) = pafScore;

		// 5. A & B already assigned to different people: Merge people if key point intersection is null
		else {
			const int assignFst = aAssign < bAssign ? aAssign : bAssign;
			const int assignSec = aAssign < bAssign ? bAssign : aAssign;
			Person2D& personFst = persons[assignFst];
			const Person2D& personSec = persons[assignSec];

			bool conflict = false;
			for (int jIdx = 0; jIdx < GetSkelDef().jointSize && !conflict; jIdx++)
				conflict |= (personFst.joints(2, jIdx) > 0.f && personSec.joints(2, jIdx) > 0.f);
				//若未出现两个“人”的某一类点都存在的情况，说明无冲突（这两个“人”是一个人的两部分）

			if (!conflict) {//合并一个人的两部分
				for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
					if (personSec.joints(2, jIdx) > 0.f)
						personFst.joints.col(jIdx) = personSec.joints.col(jIdx);

				persons.erase(persons.begin() + assignSec);
				for (Eigen::VectorXi& tmp : assignMap) {
					for (int i = 0; i < tmp.size(); i++) {
						if (tmp[i] == assignSec)
							tmp[i] = assignFst;
						else if (tmp[i] > assignSec)
							tmp[i]--;//后面的人的编号依次向前移动
					}
				}
			}
		}
	}

	// filter
	const int jcntThresh = round(0.5f * GetSkelDef().jointSize);
	for (auto person = persons.begin(); person != persons.end();) {
		if (person->GetJointCnt() < jcntThresh ) {//如果有效点还不到一半，说明这不是一个人
			const int personIdx = person - persons.begin();
			for (Eigen::VectorXi& tmp : assignMap) {
				for (int i = 0; i < tmp.size(); i++) {
					if (tmp[i] == personIdx)
						tmp[i] = -1;
					else if (tmp[i] > personIdx)
						tmp[i]--;
				}
			}
			person = persons.erase(person);//删掉这个人
		}
		else
			person++;
	}
}


void Associater::ClusterPersons2D()
{
	// cluster 2D
	for (int view = 0; view < m_cams.size(); view++) {//该帧的每一台相机
		std::vector<Eigen::VectorXi> assignMap;
		std::vector<Person2D> persons2D;
		ClusterPersons2D(m_detections[view], persons2D, assignMap);//传入的是该帧，这一台相机的数据
		m_personsMapByView[view].resize(persons2D.size(), Eigen::VectorXi::Constant(GetSkelDef().jointSize, -1));
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
			for (int candiIdx = 0; candiIdx < assignMap[jIdx].size(); candiIdx++) {
				const int pIdx = assignMap[jIdx][candiIdx];
				if (pIdx >= 0)
					m_personsMapByView[view][pIdx][jIdx] = candiIdx;//某个视图下，某个人，某一类点是哪一个
			}
		}
	}
}


void Associater::ProposalCollocation()
{
	// proposal persons
	std::function<void(const Eigen::VectorXi&, const int&, std::vector<Eigen::VectorXi>&)> Proposal
		= [&Proposal](const Eigen::VectorXi& candiCnt, const int& k, std::vector<Eigen::VectorXi>& proposals) {
		//k表示某个相机视图
		if (k == candiCnt.size()) {
			return;
		}
		else if (k == 0) {
			//第一个相机视图
			proposals = std::vector<Eigen::VectorXi>(candiCnt[k] + 1, Eigen::VectorXi::Constant(candiCnt.size(), -1));
			for (int i = 0; i < candiCnt[k]; i++)//i表示该视图中的某个人
				proposals[i + 1][k] = i;//第i+1个人在k视图中是第i个
			Proposal(candiCnt, k + 1, proposals);
		}
		else {
			std::vector<Eigen::VectorXi> proposalsBefore = proposals;
			for (int i = 0; i < candiCnt[k]; i++) {
				std::vector<Eigen::VectorXi> _proposals = proposalsBefore;
				for (auto&& _proposal : _proposals)
					_proposal[k] = i;
				proposals.insert(proposals.end(), _proposals.begin(), _proposals.end());
			}
			Proposal(candiCnt, k + 1, proposals);
		}
	};

	m_personProposals.clear();
	Eigen::VectorXi candiCnt(m_cams.size());//每台相机的视图中有多少个人
	for (int view = 0; view < m_cams.size(); view++)
		candiCnt[view] = int(m_personsMapByView[view].size());
	Proposal(candiCnt, 0, m_personProposals);
}



float Associater::CalcProposalLoss(const int& personProposalIdx)
{
	const Eigen::VectorXi& proposal = m_personProposals[personProposalIdx];
	bool valid = (proposal.array() >= 0).count() > 1;
	if (!valid)
		return -1.f;

	float loss = 0.f;

	std::vector<float> epiLosses;//投影线误差
	for (int viewA = 0; viewA < m_cams.size() - 1 && valid; viewA++) {
		if (proposal[viewA] == -1)
			continue;
		const Eigen::VectorXi& personMapA = m_personsMapByView[viewA][proposal[viewA]];
		//viewA相机中，取第proposal[viewA]个人
		for (int viewB = viewA + 1; viewB < m_cams.size() && valid; viewB++) {
			if (proposal[viewB] == -1)
				continue;
			const Eigen::VectorXi& personMapB = m_personsMapByView[viewB][proposal[viewB]];
			//viewB相机中，取第proposal[viewB]个人，认为这两个是同一个人的不同投影

			for (int jIdx = 0; jIdx < GetSkelDef().jointSize && valid; jIdx++) {
				if (personMapA[jIdx] == -1 || personMapB[jIdx] == -1)
					epiLosses.emplace_back(m_epiThresh);
				else {
					const float edge = m_jointEpiEdges[jIdx][viewA][viewB](personMapA[jIdx], personMapB[jIdx]);
					if (edge < 0.f)
						valid = false;
					else
						epiLosses.emplace_back(edge);
				}
			}
		}
	}
	if (!valid)
		return -1.f;

	if (epiLosses.size() > 0)
		loss += m_wEpi * std::accumulate(epiLosses.begin(), epiLosses.end(), 0.f) / float(epiLosses.size());

	// paf loss
	std::vector<float> pafLosses;//连接路径损失
	for (int view = 0; view < m_cams.size() && valid; view++) {
		if (proposal[view] == -1)
			continue;
		const Eigen::VectorXi& personMap = m_personsMapByView[view][proposal[view]];
		for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {
			const Eigen::Vector2i candi(personMap[GetSkelDef().pafDict(0, pafIdx)], personMap[GetSkelDef().pafDict(1, pafIdx)]);
			if (candi.x() >= 0 && candi.y() >= 0)
				pafLosses.emplace_back(1.f - m_detections[view].pafs[pafIdx](candi.x(), candi.y()));
				//损失 = 1 - 连接概率
			else
				pafLosses.emplace_back(1.f);
		}
	}
	if (pafLosses.size() > 0)
		loss += m_wPaf * std::accumulate(pafLosses.begin(), pafLosses.end(), 0.f) / float(pafLosses.size());
	// 总损失 = 投影线误差 + 连接路径损失
	// view loss
	loss += m_wView * (1.f - MathUtil::Welsch(m_cViewCnt, (proposal.array() >= 0).count()));
	return loss;
};


void Associater::ClusterPersons3D()
{
	m_personsMapByIdx.clear();

	// cluster 3D
	std::vector<std::pair<float, int>> losses;
	for (int personProposalIdx = 0; personProposalIdx < m_personProposals.size(); personProposalIdx++) {
		const float loss = CalcProposalLoss(personProposalIdx);
		const Eigen::VectorXi& proposal = m_personProposals[personProposalIdx];//某一个可能分配
		/*for (int ii = 0; ii < proposal.size(); ii++)
			std::cout << proposal[ii] << " ";
		std::cout << std::endl;*/
		if (loss > 0.f)
			losses.emplace_back(std::make_pair(loss, personProposalIdx));
	}

	// parse to cluster greedily
	std::sort(losses.begin(), losses.end());
	std::vector<Eigen::VectorXi> availableMap(m_cams.size());
	for (int view = 0; view < m_cams.size(); view++)
		availableMap[view] = Eigen::VectorXi::Constant(m_personsMapByView[view].size(), 1);//设置为未被分配（1）
		//记录某个视图中的所有2D人是否已被分配

	for (const auto& loss : losses) {
		const Eigen::VectorXi& personProposal = m_personProposals[loss.second];//某一个可能分配

		bool available = true;
		for (int i = 0; i < personProposal.size() && available; i++)
			available &= (personProposal[i] == -1 || availableMap[i][personProposal[i]]);
		//在该种假设下，存在一个视图，这个图里面有这个人并且已被分配，那么说明这种分配是不可用的

		if (!available)
			continue;

		std::vector<Eigen::VectorXi> personMap(m_cams.size(), Eigen::VectorXi::Constant(GetSkelDef().jointSize, -1));
		for (int view = 0; view < m_cams.size(); view++)
			if (personProposal[view] != -1) {
				personMap[view] = m_personsMapByView[view][personProposal[view]];//该种假设下，这个视图中这个人的连接表
				for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
					const int candiIdx = personMap[view][jIdx];
					if (candiIdx >= 0)
						m_assignMap[view][jIdx][candiIdx] = m_personsMapByIdx.size();//某个视图，某类点，某个点属于某个3D人
				}
				availableMap[view][personProposal[view]] = false;//这个视图下的这个人已被分配
			}
		m_personsMapByIdx.emplace_back(personMap);
	}

	std::cout << "personsMap_size: " << m_personsMapByIdx.size() << std::endl;

	// 有可能未被分配全，则将其他可能存在的人也加入
	// add remain persons
	for (int view = 0; view < m_cams.size(); view++) {
		for (int i = 0; i < m_personsMapByView[view].size(); i++) {
			if (availableMap[view][i]) {
				std::vector<Eigen::VectorXi> personMap(m_cams.size(), Eigen::VectorXi::Constant(GetSkelDef().jointSize, -1));
				personMap[view] = m_personsMapByView[view][i];
				m_personsMapByIdx.emplace_back(personMap);
			}
		}
	}
}


void Associater::ConstructPersons()
{
	// 2D
	// 把每个视图中，每个人、每类点对应的哪个点，以及路径进行标记
	for (int view = 0; view < m_cams.size(); view++) {
		m_persons2D[view].clear();
		for (int pIdx = 0; pIdx < m_personsMapByIdx.size(); pIdx++) {
			const std::vector<Eigen::VectorXi>& personMap = *std::next(m_personsMapByIdx.begin(),pIdx);
			const SkelDetection& detection = m_detections[view];
			Person2D person;
			for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
				if (personMap[view][jIdx] != -1)
					person.joints.col(jIdx) = detection.joints[jIdx].col(personMap[view][jIdx]);

			for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {
				const int jaIdx = GetSkelDef().pafDict(0, pafIdx);
				const int jbIdx = GetSkelDef().pafDict(1, pafIdx);
				if (personMap[view][jaIdx] != -1 && personMap[view][jbIdx] != -1)
					person.pafs[pafIdx] = detection.pafs[pafIdx](personMap[view][jaIdx], personMap[view][jbIdx]);
			}
			m_persons2D[view].emplace_back(person);
		}
	}

	// 3D
	m_persons3D.clear();
	for (int personIdx = 0; personIdx < m_personsMapByIdx.size(); personIdx++) {
		Person3D person;
		const std::vector<Eigen::VectorXi>& personMap = *std::next(m_personsMapByIdx.begin(), personIdx);
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
			MathUtil::Triangulator triangulator;
			for (int camIdx = 0; camIdx < m_cams.size(); camIdx++) {
				if (personMap[camIdx][jIdx] != -1) {
					triangulator.projs.emplace_back(m_cams[camIdx].proj);
					triangulator.points.emplace_back(m_persons2D[camIdx][personIdx].joints.col(jIdx).head(2));
				}
			}
			triangulator.Solve();
			if (triangulator.loss < m_triangulateThresh)
				person.joints.col(jIdx) = triangulator.pos.homogeneous();
			else
				person.joints.col(jIdx).setZero();
		}
		m_persons3D.emplace_back(person);
	}
}
