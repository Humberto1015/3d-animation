#include "mesh.h"
#include "MIQP.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Geometry>
#include <iostream>
#include <queue>
#include <cmath>

class ACAP{
private:
    Mesh* baseMesh;
    Mesh* targetMesh;
    std::vector<Eigen::Matrix3d> transforms;

    std::vector<Eigen::Matrix3d> polarDecomposition(const Eigen::Matrix3d&);
    void optimizeRotation(std::vector<double>&, std::vector<Eigen::Vector3d>&);
    double measureOrien(const Eigen::Vector3d&, const Eigen::Vector3d&);
    void BFS(const std::vector<Eigen::Vector3d>&, std::vector<int>&);
    void solveOrientation(const std::vector<Eigen::Vector3d>&, std::vector<int>&);
    void solveCycle(const std::vector<double>&, const std::vector<int>&, std::vector<int>&);
public:
    ACAP(Mesh&);
    void setTargetMesh(Mesh&);
    void solveTransform();
    std::vector<double> solveFeature();
    Eigen::MatrixXd solveRecon(const std::vector<double>&);
};
