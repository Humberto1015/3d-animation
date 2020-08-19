#include "ACAP.h"


ACAP::ACAP(Mesh* mesh, bool use_quat){
    this->baseMesh = mesh;
    this->use_quaternion = use_quat;
}

void ACAP::setTargetMesh(Mesh* target){
    this->targetMesh = target;
}

void ACAP::solveTransform(){

    auto num_verts = this->baseMesh->verts.rows();

    std::vector<Eigen::Matrix3d> T_matrices(num_verts);

    // for each vertex vi, compute Ti
    for (int i = 0; i < num_verts; ++i){
        int num_edges = this->baseMesh->neighbors[i].size();
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3 * num_edges, 9);
        Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(3 * num_edges, 3 * num_edges);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(3 * num_edges);
        Eigen::VectorXd solution;

        // We want to solve t11, t12, t13, t21, ......, t32, t33
        // The shape of A matrix is 3n * 9 and the shape of b vector is 3n * 1
        // n denotes the degree number of vi

        for (int j = 0; j < num_edges; ++j){
            int v_j = this->baseMesh->neighbors[i][j];

            Eigen::Vector3d e_ij = this->baseMesh->verts.row(i) - this->baseMesh->verts.row(v_j);
            Eigen::Vector3d e_ij_prime = this->targetMesh->verts.row(i) - this->targetMesh->verts.row(v_j);

            // Filling the A matrix
            A(3 * j, 0) = e_ij(0);
            A(3 * j, 3) = e_ij(1);
            A(3 * j, 6) = e_ij(2);
            A(3 * j + 1, 1) = e_ij(0);
            A(3 * j + 1, 4) = e_ij(1);
            A(3 * j + 1, 7) = e_ij(2);
            A(3 * j + 2, 2) = e_ij(0);
            A(3 * j + 2, 5) = e_ij(1);
            A(3 * j + 2, 8) = e_ij(2);

            // Filling the b vector
            b(j * 3) = e_ij_prime(0);
            b(j * 3 + 1) = e_ij_prime(1);
            b(j * 3 + 2) = e_ij_prime(2);

            // Filling the matrix of cotangent weights
            double w_ij = this->baseMesh->cot_weights[i][j];
            if (w_ij < 0)
                w_ij *= -1;

            weights(j * 3, j * 3) = sqrt(w_ij);
            weights(j * 3 + 1, j * 3 + 1) = sqrt(w_ij);
            weights(j * 3 + 2, j * 3 + 2) = sqrt(w_ij);
        }

        // Solve the least-square problem using the OR decomposition
        solution = (weights * A).colPivHouseholderQr().solve(weights * b);
        Eigen::Matrix3d T;
        for (int m = 0; m < 3; m++) {
            for (int n = 0; n < 3; n++) {
                T(m, n) = solution(m * 3 + n);
            }
        }
        T_matrices[i] = T.transpose();
    }
    this->transforms = T_matrices;
}

std::vector<double> ACAP::solveFeature(){

    // number of 1-rings
    int n = baseMesh->verts.rows();

    std::vector<double> feat(n * 9);

    if (this->use_quaternion)
        feat.resize(n * 10);


    std::vector<Eigen::Vector3d> axis(n);
    std::vector<double> angles(n);

    std::vector<Eigen::Matrix3d> R(n);
    std::vector<Eigen::Matrix3d> S(n);

    for (int i = 0; i < n; ++i){
        auto decom = polarDecomposition(this->transforms[i]);
        Eigen::AngleAxisd angleAxis(decom[0]);
        angles[i] = angleAxis.angle();
        axis[i] = angleAxis.axis();
        S[i] = decom[1];
    }

    // To do
    // step 1. solve the optimal axis set here
    AxisSolver orienSolver;
    orienSolver.solve(axis, this->baseMesh->neighbors);

    auto oriens = orienSolver.solution;
    for (int i = 0; i < axis.size(); ++i)
        axis[i] *= oriens[i];

    // step 2. solve the optimal angle set here
    AngleSolver angleSolver;
    angleSolver.solve(angles, oriens, this->baseMesh->neighbors);
    for (int i = 0; i < angles.size(); ++i){
        std::cout << angleSolver.solution[i] << "\n";
        angles[i] = 2 * M_PI * angleSolver.solution[i] + oriens[i] * angles[i];
    }

    int idx = 0;
    // set the quaternion representation of the first 1-ring
    Eigen::AngleAxisd V_0(angles[0], axis[0]);
    Eigen::Quaterniond Q_0(V_0);

    for (int i = 0; i < n; ++i){
        Eigen::AngleAxisd V_i(angles[i], axis[i]);
        Eigen::Quaterniond Q_i(V_i);

        // cancel out the global rotation to make all meshes have the same orientation
        Q_i = Q_0.inverse() * Q_i;

        // debug:
        //printf("%f %f %f %f\n", Q_i.w(), Q_i.x(), Q_i.y(), Q_i.z());

        // convert quaternion to rotation vector
        Eigen::AngleAxisd V_i_prime(Q_i);
        auto rotVec = V_i_prime.angle() * V_i_prime.axis();

        // option 1. take rotation vector as feature
        if (!this->use_quaternion){
            feat[idx++] = rotVec(0);
            feat[idx++] = rotVec(1);
            feat[idx++] = rotVec(2);
        }

        // option 2. take quaternion as feature (under experiments)
        else{
            feat[idx++] = Q_i.w();
            feat[idx++] = Q_i.x();
            feat[idx++] = Q_i.y();
            feat[idx++] = Q_i.z();
        }
        

        feat[idx++] = S[i](0, 0);
        feat[idx++] = S[i](0, 1);
        feat[idx++] = S[i](0, 2);
        feat[idx++] = S[i](1, 1);
        feat[idx++] = S[i](1, 2);
        feat[idx++] = S[i](2, 2);
    }

    std::cout << "[info] the length of ACAP feature = " << feat.size() << "\n";

    return feat;
}

Eigen::MatrixXd ACAP::solveRecon(const std::vector<double>& feat){

    int num_verts = this->baseMesh->verts.rows();

    // Step 1. Convert the ACAP features to an array of affine matrices
    std::vector<Eigen::Matrix3d> R;
    std::vector<Eigen::Matrix3d> S;
    std::vector<Eigen::Matrix3d> affines(num_verts);
    int idx = 0;
    for (int i = 0; i < affines.size(); ++i){
        Eigen::Matrix3d R_i = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d S_i = Eigen::Matrix3d::Zero();

        // option 1. recover from logR-based representation
        if (!this->use_quaternion){
            double theta_x = feat[idx++];
            double theta_y = feat[idx++];
            double theta_z = feat[idx++];
            R_i(2, 1) = theta_x;
            R_i(1, 2) = -theta_x;
            R_i(0, 2) = theta_y;
            R_i(2, 0) = -theta_y;
            R_i(1, 0) = theta_z;
            R_i(0, 1) = -theta_z;
            R_i = R_i.exp();
        }
        // option 2. recover from quaternion-based representation
        else{
            auto q_w = feat[idx++];
            auto q_x = feat[idx++];
            auto q_y = feat[idx++];
            auto q_z = feat[idx++];
            Eigen::Quaterniond Q_i(q_w, q_x, q_y, q_z);
            R_i = Q_i.matrix();
        }

        //Eigen::AngleAxisd r_x(theta_x, Eigen::Vector3d(1, 0, 0));
        //Eigen::AngleAxisd r_y(theta_y, Eigen::Vector3d(0, 1, 0));
        //Eigen::AngleAxisd r_z(theta_z, Eigen::Vector3d(0, 0, 1));

        R.emplace_back(R_i);

        S_i(0, 0) = feat[idx++];
        S_i(0, 1) = S_i(1, 0) = feat[idx++];
        S_i(0, 2) = S_i(2, 0) = feat[idx++];
        S_i(1, 1) = feat[idx++];
        S_i(1, 2) = S_i(2, 1) = feat[idx++];
        S_i(2, 2) = feat[idx++];

        S.emplace_back(S_i);

        affines[i] = R_i * S_i;
    }


    // Step 2. Solve the least suqare system
    Eigen::SparseMatrix<double> A(3 * num_verts, 3 * num_verts);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(3 * num_verts);
    Eigen::VectorXd x;

    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;

    // build the least-square system
    for (int i = 0; i < num_verts; ++i){
        // fill A
        double cotSum = 0;
        for (auto c_ij: this->baseMesh->cot_weights[i])
            cotSum += c_ij;

        tripletList.emplace_back(T(3 * i, 3 * i, cotSum));
        tripletList.emplace_back(T(3 * i + 1, 3 * i + 1, cotSum));
        tripletList.emplace_back(T(3 * i + 2, 3 * i + 2, cotSum));

        for (int j = 0; j < this->baseMesh->neighbors[i].size(); ++j){
            int vj_index = this->baseMesh->neighbors[i][j];
            double c_ij = this->baseMesh->cot_weights[i][j];

            tripletList.emplace_back(T(3 * i, 3 * vj_index, -c_ij));
            tripletList.emplace_back(T(3 * i + 1, 3 * vj_index + 1, -c_ij));
            tripletList.emplace_back(T(3 * i + 2, 3 * vj_index + 2, -c_ij));
        }
        // fill b
        Eigen::VectorXd term = Eigen::VectorXd::Zero(3);
        for (int j = 0; j < this->baseMesh->neighbors[i].size(); ++j){
            int vj_index = this->baseMesh->neighbors[i][j];
            double c_ij = this->baseMesh->cot_weights[i][j];
            Eigen::Vector3d q_j = this->baseMesh->verts.row(vj_index);
            Eigen::Vector3d q_i = this->baseMesh->verts.row(i);
            term += c_ij * (affines[i] + affines[vj_index]) * (q_i - q_j);
        }
        b(3 * i) = term(0);
        b(3 * i + 1) = term(1);
        b(3 * i + 2) = term(2);
    }
    A.setFromTriplets(tripletList.begin(), tripletList.end());

    // Do Cholesky factorization on A matrix to speed up the optimization
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<double> > solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        //Decomposition failed
        std::cout << "Decomposition failed." << std::endl;
        exit(1);
    }
    x = solver.solve(b);

    Eigen::MatrixXd verts(num_verts, 3);

    for (int i = 0; i < num_verts; ++i){
        Eigen::Vector3d row;
        row(0) = x(3 * i);
        row(1) = x(3 * i + 1);
        row(2) = x(3 * i + 2);
        verts.row(i) = row;
    }
    return verts;
}

std::vector<Eigen::Matrix3d> ACAP::polarDecomposition(const Eigen::Matrix3d& T){

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(T, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U, V;
    U = svd.matrixU();
    V = svd.matrixV();
    Eigen::Matrix3d S(svd.singularValues().asDiagonal());
    Eigen::Matrix3d Temp = Eigen::Matrix3d::Identity();
    Temp(2, 2) = (U * V.transpose()).determinant();
    Eigen::Matrix3d R = U * Temp * V.transpose();
    Eigen::Matrix3d Scale = V * Temp * S * V.transpose();

    std::vector<Eigen::Matrix3d> decomposition;
    decomposition.emplace_back(R);
    decomposition.emplace_back(Scale);

    return decomposition;
}