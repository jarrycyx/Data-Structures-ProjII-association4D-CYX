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
			//-1��ʾδ����ͶӰ2D����


	for (int view = 0; view < m_cams.size(); view++) {
		m_persons2D[view].clear();
		m_personsMapByView[view].clear();
	}
	m_persons3D.clear();

}


void Associater::ConstructJointRays()
{
	for (int view = 0; view < m_cams.size(); view++) {
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {//����������ܵ�ÿһ���
			const Eigen::Matrix3Xf& joints = m_detections[view].joints[jIdx];
			m_jointRays[view][jIdx].resize(3, joints.cols());//m_jointRays�洢ÿ����ά��������ά����
			for (int jCandiIdx = 0; jCandiIdx < joints.cols(); jCandiIdx++)//����һ֡�У��ҳ������ɸ�ĳһ���
				m_jointRays[view][jIdx].col(jCandiIdx) = m_cams[view].CalcRay(joints.block<2, 1>(0, jCandiIdx));
				//����һ̨�����һ�����ÿ�����3DͶӰ����
				//3D�е�ʵ�ʵ�����ڹ�����ͶӰ���������ϵ�����λ��
		}
	}
}


void Associater::ConstructJointEpiEdges()
{
	for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
		for (int viewA = 0; viewA < m_cams.size() - 1; viewA++) {
			for (int viewB = viewA + 1; viewB < m_cams.size(); viewB++) {//�����������ڵ����
				Eigen::MatrixXf& epi = m_jointEpiEdges[jIdx][viewA][viewB];//��A��B��̨����лָ��ĵ�ֵ3D����
				const Eigen::Matrix3Xf& jointsA = m_detections[viewA].joints[jIdx];
				const Eigen::Matrix3Xf& jointsB = m_detections[viewB].joints[jIdx];//ԭʼ������
				const Eigen::Matrix3Xf& raysA = m_jointRays[viewA][jIdx];
				const Eigen::Matrix3Xf& raysB = m_jointRays[viewB][jIdx];//��ֵ��һ��ͶӰֱ�ߣ��Ľ�������
				epi.setConstant(jointsA.cols(), jointsB.cols(), -1.f);//����ΪA����м�⵽ĳһ������Ŀ������ΪB����м�⵽ĳһ������Ŀ
				for (int jaCandiIdx = 0; jaCandiIdx < epi.rows(); jaCandiIdx++) {
					for (int jbCandiIdx = 0; jbCandiIdx < epi.cols(); jbCandiIdx++) {//��̨�����ĳһ�������е㶼Ҫ����
						const float dist = MathUtil::Line2LineDist(//���߼�࣬ÿ������������ĺ�3DԤ��ͶӰ�������
							m_cams[viewA].pos, raysA.col(jaCandiIdx), m_cams[viewB].pos, raysB.col(jbCandiIdx));
						if (dist < m_epiThresh)
							epi(jaCandiIdx, jbCandiIdx) = dist / m_epiThresh;
					}
				}
				m_jointEpiEdges[jIdx][viewB][viewA] = epi.transpose();//�洢ĳһ��㣬����������������е�ͶӰ�ߵľ���
			}
		}
	}
}



void Associater::ClusterPersons2D(const SkelDetection& detection, std::vector<Person2D>& persons, std::vector<Eigen::VectorXi>& assignMap)
{
	persons.clear();

	// generate valid pafs
	std::vector<std::tuple<float, int, int, int>> pafSet;
	for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {//�������ģ�͵�ÿһ�����ӱ�
		const int jaIdx = GetSkelDef().pafDict(0, pafIdx);//�����ߵ�����������
		const int jbIdx = GetSkelDef().pafDict(1, pafIdx);
		for (int jaCandiIdx = 0; jaCandiIdx < detection.joints[jaIdx].cols(); jaCandiIdx++) {//̽�⵽�����������Ķ���
			for (int jbCandiIdx = 0; jbCandiIdx < detection.joints[jbIdx].cols(); jbCandiIdx++) {
				const float jaScore = detection.joints[jaIdx](2, jaCandiIdx);//�ö����Ӧ�ĸ���
				const float jbScore = detection.joints[jbIdx](2, jbCandiIdx);
				const float pafScore = detection.pafs[pafIdx](jaCandiIdx, jbCandiIdx);
				if (jaScore > 0.f && jbScore > 0.f && pafScore > 0.f)
					pafSet.emplace_back(std::make_tuple(pafScore, pafIdx, jaCandiIdx, jbCandiIdx));
					//������У�ĳ������������ϵ
			}
		}
	}
	std::sort(pafSet.rbegin(), pafSet.rend());//���б�����

	// construct bodies use minimal spanning tree
	assignMap.resize(GetSkelDef().jointSize);
	for (int jIdx = 0; jIdx < assignMap.size(); jIdx++)
		assignMap[jIdx].setConstant(detection.joints[jIdx].cols(), -1);

	for (const auto& paf : pafSet) {
		const float pafScore = std::get<0>(paf);
		const int pafIdx = std::get<1>(paf);//��������бߵı��
		const int jaCandiIdx = std::get<2>(paf);//ĳ����У�ĳ��������
		const int jbCandiIdx = std::get<3>(paf);
		const int jaIdx = GetSkelDef().pafDict(0, pafIdx);//ĳ����ڹ�������еı��
		const int jbIdx = GetSkelDef().pafDict(1, pafIdx);

		int& aAssign = assignMap[jaIdx][jaCandiIdx];//ĳ����ĳ��������ĳ����
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
				//��δ�����������ˡ���ĳһ��㶼���ڵ������˵���޳�ͻ�����������ˡ���һ���˵������֣�

			if (!conflict) {//�ϲ�һ���˵�������
				for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
					if (personSec.joints(2, jIdx) > 0.f)
						personFst.joints.col(jIdx) = personSec.joints.col(jIdx);

				persons.erase(persons.begin() + assignSec);
				for (Eigen::VectorXi& tmp : assignMap) {
					for (int i = 0; i < tmp.size(); i++) {
						if (tmp[i] == assignSec)
							tmp[i] = assignFst;
						else if (tmp[i] > assignSec)
							tmp[i]--;//������˵ı��������ǰ�ƶ�
					}
				}
			}
		}
	}

	// filter
	const int jcntThresh = round(0.5f * GetSkelDef().jointSize);
	for (auto person = persons.begin(); person != persons.end();) {
		if (person->GetJointCnt() < jcntThresh ) {//�����Ч�㻹����һ�룬˵���ⲻ��һ����
			const int personIdx = person - persons.begin();
			for (Eigen::VectorXi& tmp : assignMap) {
				for (int i = 0; i < tmp.size(); i++) {
					if (tmp[i] == personIdx)
						tmp[i] = -1;
					else if (tmp[i] > personIdx)
						tmp[i]--;
				}
			}
			person = persons.erase(person);//ɾ�������
		}
		else
			person++;
	}
}


void Associater::ClusterPersons2D()
{
	// cluster 2D
	for (int view = 0; view < m_cams.size(); view++) {//��֡��ÿһ̨���
		std::vector<Eigen::VectorXi> assignMap;
		std::vector<Person2D> persons2D;
		ClusterPersons2D(m_detections[view], persons2D, assignMap);//������Ǹ�֡����һ̨���������
		m_personsMapByView[view].resize(persons2D.size(), Eigen::VectorXi::Constant(GetSkelDef().jointSize, -1));
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
			for (int candiIdx = 0; candiIdx < assignMap[jIdx].size(); candiIdx++) {
				const int pIdx = assignMap[jIdx][candiIdx];
				if (pIdx >= 0)
					m_personsMapByView[view][pIdx][jIdx] = candiIdx;//ĳ����ͼ�£�ĳ���ˣ�ĳһ�������һ��
			}
		}
	}
}


void Associater::ProposalCollocation()
{
	// proposal persons
	std::function<void(const Eigen::VectorXi&, const int&, std::vector<Eigen::VectorXi>&)> Proposal
		= [&Proposal](const Eigen::VectorXi& candiCnt, const int& k, std::vector<Eigen::VectorXi>& proposals) {
		//k��ʾĳ�������ͼ
		if (k == candiCnt.size()) {
			return;
		}
		else if (k == 0) {
			//��һ�������ͼ
			proposals = std::vector<Eigen::VectorXi>(candiCnt[k] + 1, Eigen::VectorXi::Constant(candiCnt.size(), -1));
			for (int i = 0; i < candiCnt[k]; i++)//i��ʾ����ͼ�е�ĳ����
				proposals[i + 1][k] = i;//��i+1������k��ͼ���ǵ�i��
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
	Eigen::VectorXi candiCnt(m_cams.size());//ÿ̨�������ͼ���ж��ٸ���
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

	std::vector<float> epiLosses;//ͶӰ�����
	for (int viewA = 0; viewA < m_cams.size() - 1 && valid; viewA++) {
		if (proposal[viewA] == -1)
			continue;
		const Eigen::VectorXi& personMapA = m_personsMapByView[viewA][proposal[viewA]];
		//viewA����У�ȡ��proposal[viewA]����
		for (int viewB = viewA + 1; viewB < m_cams.size() && valid; viewB++) {
			if (proposal[viewB] == -1)
				continue;
			const Eigen::VectorXi& personMapB = m_personsMapByView[viewB][proposal[viewB]];
			//viewB����У�ȡ��proposal[viewB]���ˣ���Ϊ��������ͬһ���˵Ĳ�ͬͶӰ

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
	std::vector<float> pafLosses;//����·����ʧ
	for (int view = 0; view < m_cams.size() && valid; view++) {
		if (proposal[view] == -1)
			continue;
		const Eigen::VectorXi& personMap = m_personsMapByView[view][proposal[view]];
		for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {
			const Eigen::Vector2i candi(personMap[GetSkelDef().pafDict(0, pafIdx)], personMap[GetSkelDef().pafDict(1, pafIdx)]);
			if (candi.x() >= 0 && candi.y() >= 0)
				pafLosses.emplace_back(1.f - m_detections[view].pafs[pafIdx](candi.x(), candi.y()));
				//��ʧ = 1 - ���Ӹ���
			else
				pafLosses.emplace_back(1.f);
		}
	}
	if (pafLosses.size() > 0)
		loss += m_wPaf * std::accumulate(pafLosses.begin(), pafLosses.end(), 0.f) / float(pafLosses.size());
	// ����ʧ = ͶӰ����� + ����·����ʧ
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
		const Eigen::VectorXi& proposal = m_personProposals[personProposalIdx];//ĳһ�����ܷ���
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
		availableMap[view] = Eigen::VectorXi::Constant(m_personsMapByView[view].size(), 1);//����Ϊδ�����䣨1��
		//��¼ĳ����ͼ�е�����2D���Ƿ��ѱ�����

	for (const auto& loss : losses) {
		const Eigen::VectorXi& personProposal = m_personProposals[loss.second];//ĳһ�����ܷ���

		bool available = true;
		for (int i = 0; i < personProposal.size() && available; i++)
			available &= (personProposal[i] == -1 || availableMap[i][personProposal[i]]);
		//�ڸ��ּ����£�����һ����ͼ�����ͼ����������˲����ѱ����䣬��ô˵�����ַ����ǲ����õ�

		if (!available)
			continue;

		std::vector<Eigen::VectorXi> personMap(m_cams.size(), Eigen::VectorXi::Constant(GetSkelDef().jointSize, -1));
		for (int view = 0; view < m_cams.size(); view++)
			if (personProposal[view] != -1) {
				personMap[view] = m_personsMapByView[view][personProposal[view]];//���ּ����£������ͼ������˵����ӱ�
				for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
					const int candiIdx = personMap[view][jIdx];
					if (candiIdx >= 0)
						m_assignMap[view][jIdx][candiIdx] = m_personsMapByIdx.size();//ĳ����ͼ��ĳ��㣬ĳ��������ĳ��3D��
				}
				availableMap[view][personProposal[view]] = false;//�����ͼ�µ�������ѱ�����
			}
		m_personsMapByIdx.emplace_back(personMap);
	}

	std::cout << "personsMap_size: " << m_personsMapByIdx.size() << std::endl;

	// �п���δ������ȫ�����������ܴ��ڵ���Ҳ����
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
	// ��ÿ����ͼ�У�ÿ���ˡ�ÿ����Ӧ���ĸ��㣬�Լ�·�����б��
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
