#include "mesh.h"
#include "axisSolver.h"
#include "angleSolver.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Geometry>
#include <iostream>
#include <queue>
#include <cmath>
#include <igl/readPLY.h>


class ACAP{
private:
    Mesh* baseMesh;
    Mesh* targetMesh;
    std::vector<Eigen::Matrix3d> transforms;

    // for debug
    bool use_quaternion = false;

    // parameters: a transformation matrix of certain 1-ring
    // output: 
    // (1) a rotation matrix
    // (2) a scaling/shearing matrix
    std::vector<Eigen::Matrix3d> polarDecomposition(const Eigen::Matrix3d&);
public:
    // Constructor
    // parameters: a pointer that points to the reference mesh object
    ACAP(Mesh*, bool);

    // parameters: a pointer that points to the target mesh object
    void setTargetMesh(Mesh*);

    // parameters: None
    // this member function solves transformations for all 1-rings
    // and saves the result in member variable "transforms"
    void solveTransform();

    // parameters: None
    // this member function solves the ACAP feature from transformation matrices
    // and return the results as a vector
    std::vector<double> solveFeature();

    // parameters: the ACAP feature vector
    // this member function recovers the mesh from given ACAP feature vector
    Eigen::MatrixXd solveRecon(const std::vector<double>&);
};
