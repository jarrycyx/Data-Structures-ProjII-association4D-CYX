#pragma once
#include "skel.h"
#include "camera.h"


class Associater
{
public:
	Associater(const std::vector<Camera>& _cams);
	~Associater() = default;
	Associater(const Associater& _) = delete;
	Associater& operator=(const Associater& _) = delete;


	const std::vector<std::vector<Person2D>>& GetPersons2D() const { return m_persons2D; }
	const std::vector<Person3D>& GetPersons3D() const { return m_persons3D; }

	void SetDetection(const std::vector<SkelDetection>& detections);
	void ConstructJointRays();
	void ConstructJointEpiEdges();
	void ClusterPersons2D(const SkelDetection& detection, std::vector<Person2D>& persons, std::vector<Eigen::VectorXi>& assignMap);
	void ClusterPersons2D();
	void ClusterPersons3D();
	void ProposalCollocation();
	float CalcProposalLoss(const int& personProposalIdx);
	void ConstructPersons();

private:
	float m_epiThresh;
	float m_wEpi;
	float m_wView;
	float m_wPaf;
	float m_cPlaneTheta;
	float m_cViewCnt;
	float m_triangulateThresh;

	std::vector<Camera> m_cams;
	std::vector<SkelDetection> m_detections;
	std::vector<std::vector<Eigen::Matrix3Xf>> m_jointRays;
	std::vector<std::vector<std::vector<Eigen::MatrixXf>>> m_jointEpiEdges;	// m_epiEdge[jIdx][viewA][viewB](jA, jB) = epipolar distance
	std::vector<std::vector<Eigen::VectorXi>> m_personsMapByIdx, m_personsMapByView;
	std::vector<Eigen::VectorXi> m_personProposals;
	std::vector<std::vector<Eigen::VectorXi>> m_assignMap;
	std::vector<std::vector<Person2D>> m_persons2D;
	std::vector<Person3D> m_persons3D;
};

