#include "mesh.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <queue>


class RIMD{
public:

    Mesh* baseMesh;
    Mesh* targetMesh;

    std::vector<Eigen::Matrix3d> transforms;
    std::vector<std::vector<Eigen::Matrix3d> > features;

    RIMD(Mesh& base, Mesh& target){
        this->baseMesh = &base;
        this->targetMesh = &target;
        solveTransform();
        buildRIMD();
    }

    // given RIMD features, reconstruct the target mesh
    Eigen::MatrixXd solveReconstruct(const std::vector<std::vector<Eigen::Matrix3d> >& rimd_vectors){

        std::cout << "[ Start to reconstruct the mesh from RIMD representations ]" << std::endl;

        // build the least-square system
    	// n denotes the vertex number
    	// Aj is a 3 * 3n matrix
    	// A = [A0 A1 A2 ... An]'
    	// So the size of A is 3n * 3n
        std::cout << "- Initialize the A matrix..." << " ";
    	int num_verts = rimd_vectors.size();
    	Eigen::SparseMatrix<double> A(3 * num_verts, 3 * num_verts);
    	typedef Eigen::Triplet<double> T;
    	std::vector<T> tripletList;
    	// Build the A matrix
    	// Traverse all vertices
    	for (int j = 0; j < num_verts; j++) {
    		// Traverse all neighbors adjacent to vj

    		// Compute the sum of cotangent weights of vj
    		double sum_of_weights = 0;
    		for (int k = 0; k < this->baseMesh->neighbors[j].size(); k++) {
    			sum_of_weights += this->baseMesh->cot_weights[j][k];
    		}

    		tripletList.emplace_back(T(3 * j, 3 * j, sum_of_weights));
    		tripletList.emplace_back(T(3 * j + 1, 3 * j + 1, sum_of_weights));
    		tripletList.emplace_back(T(3 * j + 2, 3 * j + 2, sum_of_weights));

    		for (int k = 0; k < this->baseMesh->neighbors[j].size(); k++) {
    			int vk_index = this->baseMesh->neighbors[j][k];
    			double c_jk = this->baseMesh->cot_weights[j][k];

    			tripletList.emplace_back(T(3 * j, 3 * vk_index, -c_jk));
    			tripletList.emplace_back(T(3 * j + 1, 3 * vk_index + 1, -c_jk));
    			tripletList.emplace_back(T(3 * j + 2, 3 * vk_index + 2, -c_jk));
    		}
    	}
    	A.setFromTriplets(tripletList.begin(), tripletList.end());

        // Do Cholesky factorization on the A matrix
    	Eigen::SimplicialCholesky<Eigen::SparseMatrix<double> > solver;
    	solver.compute(A);
    	if (solver.info() != Eigen::Success) {
    		//Decomposition failed
    		std::cout << "Decomposition failed." << std::endl;
    		exit(1);
    	}

    	Eigen::VectorXd b(3 * num_verts);
    	Eigen::VectorXd x;
        std::cout << "Done." << std::endl;


        // initialize rotation matrices
        std::cout << "- Initialize the rotation matrices..." << " ";
        std::vector<Eigen::Matrix3d> R;
    	// Initial all R matrices using the BFS method
        BFS_initialize(rimd_vectors, R);

        //R.resize(num_verts);
        //for (int i = 0; i < R.size(); i++) {
    	//	R[i].setIdentity();
    	//}
        std::cout << "Done." << std::endl;

        // Run iteration!!!
    	int max_iter = 200;
    	int iter = 0;
        double epsilon = 0.001;
        double loss, loss_prev, delta;
        loss_prev = 0;
        delta = 100;

        while (iter <= max_iter && delta > epsilon){
            iter++;
            printf("[Iteration: %d]\n", iter);
            // Fixed rotation matrices, solve for point positions
            solveGlobalStep(solver, x, b, R, rimd_vectors);
            // Fixed point positions, update rotation matrices
            solveLocalStep(x, R, rimd_vectors);

            loss = computeEnergy(x, R, rimd_vectors);

            delta = abs((loss - loss_prev));
            loss_prev = loss;

            printf("Loss = %f\n", loss);

            // debug
            //break;
        }

        // save the point positions
        Eigen::MatrixXd verts_recon;
        verts_recon.resize(num_verts, 3);
        for (int i = 0; i < num_verts; ++i) {
            Eigen::Vector3d v;
            v(0) = x(3 * i);
            v(1) = x(3 * i + 1);
            v(2) = x(3 * i + 2);
            verts_recon.row(i) = v;
        }

        return verts_recon;
    }

    // if the RIMD feature is directly derived from the target mesh, run it.
    Eigen::MatrixXd quickReconstruct(const std::vector<std::vector<Eigen::Matrix3d>>& rimd_vectors) {
    	// Ti * e_ij = e'_ij
    	int num_verts = this->baseMesh->verts.rows();
    	Eigen::MatrixXd P;
        P.resize(num_verts, 3);
    	Eigen::MatrixXd P_prime;
        P_prime.resize(num_verts, 3);
    	for (int i = 0; i < num_verts; i++) {
    		P.row(i) = this->baseMesh->verts.row(i);
    	}

    	// Create a list of R matrices
    	std::vector<Eigen::Matrix3d> R(num_verts);
    	// Create the color list which denotes the state of visitation
    	std::vector<int> Color(num_verts);
    	// Create the parent list
    	std::vector<int> Parent(num_verts);

    	// Define some constant
    	int white = -2; // undiscovered
    	int gray = -1; // if u is discovered, the adjacent vertex v is undiscovered
    	int black = 0; // discovered
    	int unknown = -3;

    	// Initialization
    	for (int i = 0; i < num_verts; i++) {
    		Color[i] = white;
    		Parent[i] = -1; // Parent is unknown
    	}
    	// Choose an arbitrary vertex
    	// I take the first vertex as the start vertex
    	int s = 0;
    	Color[s] = gray;
    	Parent[s] = unknown;
    	P_prime.row(s) = P.row(s);

    	// Set to the identity matrix
    	R[s].setIdentity();

    	// Create an empty queue
    	std::queue<int> Q;
    	// put the start vertex into the queue
    	Q.push(s);

    	while (!Q.empty()) {
    		int u = Q.front();
    		Q.pop();
    		for (int i = 0; i < this->baseMesh->neighbors[u].size(); i++) {

    			// Compute the initial value
    			Eigen::Matrix3d Ru = R[u];
    			Eigen::Matrix3d dRuv = rimd_vectors[u][i].exp();

    			int v = this->baseMesh->neighbors[u][i];
    			if (Color[v] == white) {
    				Color[v] = gray;
    				Parent[v] = u;
    				Q.push(v);
    				R[v] = Ru * dRuv;
    				Eigen::Vector3d evu;
    				Eigen::Vector3d temp;
    				temp = this->baseMesh->verts.row(v) - this->baseMesh->verts.row(u);
    				evu(0) = temp(0);
    				evu(1) = temp(1);
    				evu(2) = temp(2);
    				Eigen::Matrix3d Sv = rimd_vectors[v].back();
    				Eigen::Vector3d evu_prime = R[v] * Sv * evu;

    				temp(0) = evu_prime(0);
    				temp(1) = evu_prime(1);
    				temp(2) = evu_prime(2);
    				// Update vertex
    				P_prime(v, 0) = temp(0) + P_prime(u, 0);
                    P_prime(v, 1) = temp(1) + P_prime(u, 1);
                    P_prime(v, 2) = temp(2) + P_prime(u, 2);
    			}
    		}
    		Color[u] = black;
    	}

    	return P_prime;
    }


private:
    // input: base mesh and target mesh
    // output: transformations from base mesh to target mesh
    void solveTransform(){
        std::vector<Eigen::Matrix3d> T_matrices;

        int num_verts = baseMesh->verts.rows();

        // for each vertex vi, compute Ti
        for (int i = 0; i < num_verts; ++i){
            int num_edges = baseMesh->neighbors[i].size();
            Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3 * num_edges, 9);
            Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(3 * num_edges, 3 * num_edges);
            Eigen::VectorXd b = Eigen::VectorXd::Zero(3 * num_edges);
            Eigen::VectorXd solution;

            // We want to solve t11, t12, t13, t21, ......, t32, t33
		    // The shape of A matrix is 3n * 9 and the shape of b vector is 3n * 1
		    // n denotes the degree number of vi

            for (int j = 0; j < num_edges; ++j){
                int v_j = baseMesh->neighbors[i][j];

                Eigen::Vector3d e_ij = baseMesh->verts.row(i) - baseMesh->verts.row(v_j);
                Eigen::Vector3d e_ij_prime = targetMesh->verts.row(i) - targetMesh->verts.row(v_j);

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
                double w_ij = baseMesh->cot_weights[i][j];
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
		    T_matrices.emplace_back(T.transpose()); //T.transpose()
        }

        this->transforms = T_matrices;
    }

    std::vector<Eigen::Matrix3d> polarDecomposition(const Eigen::Matrix3d& T){
        // Reduce the problem to SVD decomposition.
    	Eigen::JacobiSVD<Eigen::MatrixXd> svd(T, Eigen::ComputeFullU | Eigen::ComputeFullV);
    	// T = RS

    	Eigen::MatrixXd U, V, S, R, SV;
    	U = svd.matrixU();
    	V = svd.matrixV();
    	SV = svd.singularValues();
    	R = U * V.transpose();

    	const auto & SVT = SV.asDiagonal() * V.adjoint();

    	// Check for reflection
    	if (R.determinant() < 0) {
    		auto W = V.eval();
    		W.col(V.cols() - 1) *= -1.;
    		R = U * W.transpose();
    		S = W * SVT;
    	}
    	else {
    		S = V * SVT;
    	}


    	std::vector<Eigen::Matrix3d> decomposition;
    	decomposition.emplace_back(R);
    	decomposition.emplace_back(S);


    	return decomposition;
    }

    void buildRIMD(){
        // f = {log_dR_ij; Si}
        std::vector<std::vector<Eigen::Matrix3d> > rimd_features;

        int num_verts = this->baseMesh->verts.rows();

        for (int i = 0; i < num_verts; ++i){
            Eigen::Matrix3d R_i, S_i;
            std::vector<Eigen::Matrix3d> T = polarDecomposition(this->transforms[i]);
            R_i = T[0];
            S_i = T[1];

            std::vector<Eigen::Matrix3d> feature;

            // Filling logdR_ij
    		for (int j = 0; j < this->baseMesh->neighbors[i].size(); ++j) {
    			int v_j = this->baseMesh->neighbors[i][j];
    			Eigen::Matrix3d T_j = this->transforms[v_j];
    			Eigen::Matrix3d R_j = polarDecomposition(T_j)[0];
    			Eigen::Matrix3d log_dR_ij = (R_i.transpose() * R_j).log();
    			feature.emplace_back(log_dR_ij);
    		}
    		// Filling the Si matrix
    		feature.emplace_back(S_i);
    		rimd_features.emplace_back(feature);
        }
        this->features = rimd_features;
    }

    Eigen::Matrix3d compute_ring(const std::vector<std::vector<Eigen::Matrix3d> >& rimd_vectors, const std::vector<Eigen::Matrix3d>& R, int j) {
    	Eigen::Matrix3d sum = Eigen::Matrix3d::Zero();
    	double cj_tilde = 1.0 / this->baseMesh->neighbors[j].size();
    	for (int i = 0; i < this->baseMesh->neighbors[j].size(); ++i) {
    		int vi_index = this->baseMesh->neighbors[j][i];
    		Eigen::Matrix3d Ri = R[vi_index];
    		Eigen::Matrix3d Sj = rimd_vectors[j].back();
    		int vj_position;
    		for (vj_position = 0; vj_position < this->baseMesh->neighbors[vi_index].size(); ++vj_position) {
    			if (this->baseMesh->neighbors[vi_index][vj_position] == j)
    				break;
    		}

    		Eigen::Matrix3d dRij = rimd_vectors[vi_index][vj_position].exp();

    		sum += Ri * dRij * Sj;
    	}
    	return cj_tilde * sum;
    }

    void BFS_initialize(const std::vector<std::vector<Eigen::Matrix3d>>& rimd_vectors, std::vector<Eigen::Matrix3d>& R) {

    	int num_verts = this->baseMesh->verts.rows();
    	// Create a list of R matrices
    	R.resize(num_verts);
    	// Create the color list which denotes the state of visitation
    	std::vector<int> Color(num_verts);
    	// Create the parent list
    	std::vector<int> Parent(num_verts);

    	// Define some constant
    	int white = -2; // undiscover
    	int gray = -1; // if u is discovered, the adjacent vertex v is undiscovered
    	int black = 0; // discovered
    	int unknown = -3;

    	// Initialization
    	for (int i = 0; i < num_verts; i++) {
    		Color[i] = white;
    		Parent[i] = -1; // Parent is unknown
    	}
    	// Visit the start vertex
    	// I take the first vertex as the start vertex
    	int s = 0;
    	Color[s] = gray;
    	Parent[s] = unknown;

    	// Set to the identity matrix
    	R[s].setIdentity();

    	// Create an empty queue
    	std::queue<int> Q;
    	// put the start vertex into the queue
    	Q.push(s);

    	while (!Q.empty()) {
    		int u = Q.front();
    		Q.pop();
    		for (int i = 0; i < this->baseMesh->neighbors[u].size(); i++) {

    			// Compute the initial value
    			Eigen::Matrix3d Ru = R[u];
    			Eigen::Matrix3d dRuv = rimd_vectors[u][i].exp();


    			int v = this->baseMesh->neighbors[u][i];
    			if (Color[v] == white) {
    				Color[v] = gray;
    				Parent[v] = u;
    				Q.push(v);
    				R[v] = Ru * dRuv;
    			}
    		}
    		Color[u] = black;
    	}
    }

    void solveGlobalStep(Eigen::SimplicialCholesky<Eigen::SparseMatrix<double> >& solver, Eigen::VectorXd& x, Eigen::VectorXd& b, const std::vector<Eigen::Matrix3d>& R, const std::vector<std::vector<Eigen::Matrix3d> >& rimd_vectors){
        std::cout << " Solve the global step..." << " ";

        int num_verts = R.size();
        // Traverse all vj's
		for (int j = 0; j < num_verts; j++) {
			Eigen::Vector3d sum = Eigen::Vector3d::Zero();
			// Saved for use in the second pass.
			auto first_ring_sum = compute_ring(rimd_vectors, R, j);

			for (int k = 0; k < this->baseMesh->neighbors[j].size(); k++) {
				double c_jk = this->baseMesh->cot_weights[j][k];
				int vk_index = this->baseMesh->neighbors[j][k];
                Eigen::Vector3d ejk = this->baseMesh->verts.row(j) - this->baseMesh->verts.row(vk_index);
				auto second_ring_sum = compute_ring(rimd_vectors, R, vk_index);
				sum += c_jk * (first_ring_sum + second_ring_sum) * ejk;
			}
			//sum /= 2;
			b(3 * j) = sum(0);
			b(3 * j + 1) = sum(1);
			b(3 * j + 2) = sum(2);
		}

		x = solver.solve(b);

		if (solver.info() != Eigen::Success) {
			//Solving failed
			std::cout << "Solving failed" << std::endl;
			exit(2);
		}

        std::cout << "Done." << std::endl;
    }

    void solveLocalStep(const Eigen::VectorXd& x, std::vector<Eigen::Matrix3d>& R, const std::vector<std::vector<Eigen::Matrix3d> >& rimd_vectors){
        std::cout << " Solve the local step..." << " ";

        int num_verts = R.size();
        // Traverse all vertices to update Ri
		for (int i = 0; i < num_verts; i++) {
			Eigen::Matrix3d Qi = Eigen::Matrix3d::Zero();
			// Traverse all vi's neighbors vj's
			for (int j = 0; j < this->baseMesh->neighbors[i].size(); j++) {
				// Travese all vj's neighbors vk's
				Eigen::Matrix3d second_term = Eigen::Matrix3d::Zero();
				int vj_index = this->baseMesh->neighbors[i][j];
				for (int k = 0; k < this->baseMesh->neighbors[vj_index].size(); k++) {
					double cjk = this->baseMesh->cot_weights[vj_index][k];
					// Get ejk
                    int vk_index = this->baseMesh->neighbors[vj_index][k];
					Eigen::Vector3d ejk = this->baseMesh->verts.row(vj_index) - this->baseMesh->verts.row(vk_index);
					// Get e'jk
					Eigen::Vector3d ejk_prime;
					ejk_prime(0) = x(3 * vj_index) - x(3 * vk_index);
					ejk_prime(1) = x(3 * vj_index + 1) - x(3 * vk_index + 1);
					ejk_prime(2) = x(3 * vj_index + 2) - x(3 * vk_index + 2);

					second_term += cjk * ejk * ejk_prime.transpose();
				}
				double cj_tilde = 1.0 / this->baseMesh->neighbors[vj_index].size();
				Eigen::Matrix3d dRij = rimd_vectors[i][j].exp();
				Eigen::Matrix3d Sj = rimd_vectors[vj_index].back();

				Qi += cj_tilde * dRij * Sj * second_term;
			}
			//std::cout << Qi << std::endl;
			//std::cout << "-" << std::endl;

			// Compute Ri using SVD decomposition
			Eigen::JacobiSVD<Eigen::MatrixXd> svd(Qi, Eigen::ComputeFullU | Eigen::ComputeFullV);
			Eigen::MatrixXd Ui = svd.matrixU();
			Eigen::MatrixXd Vi = svd.matrixV();
			Eigen::MatrixXd SV = svd.singularValues();
			Eigen::MatrixXd Ri = Vi * Ui.transpose(); // Ui * Vi.transpose()

			// Choose appropriate signs by flipping all elements to make det(Ri) > 0
			if (Ri.determinant() < 0) {
				Eigen::JacobiSVD<Eigen::MatrixXd> svd(Ri, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Eigen::MatrixXd U = svd.matrixU();
                Eigen::MatrixXd V = svd.matrixV();
                V.col(2) *= -1;
                Ri = V * U.transpose();
			}
			// Update the Ri matrix
			R[i] = Ri;
		}

        std::cout << "Done." << std::endl;
    }

    double computeEnergy(const Eigen::VectorXd& x, const std::vector<Eigen::Matrix3d>& R, const std::vector<std::vector<Eigen::Matrix3d> >& rimd_vectors){

        int num_verts = R.size();
		double energy = 0;
		for (int i = 0; i < num_verts; i++) {
			double second_term = 0;
			for (int j = 0; j < this->baseMesh->neighbors[i].size(); j++) {
				int vj_index = this->baseMesh->neighbors[i][j];
				double cj_tilde = 1.0 / this->baseMesh->neighbors[vj_index].size();

				double third_term = 0;
				for (int k = 0; k < this->baseMesh->neighbors[vj_index].size(); k++) {
					double cjk = this->baseMesh->cot_weights[vj_index][k];
					int vk_index = this->baseMesh->neighbors[vj_index][k];
					Eigen::Matrix3d dRij = rimd_vectors[i][j].exp();
					Eigen::Matrix3d Sj = rimd_vectors[vj_index].back();

					Eigen::Vector3d ejk, ejk_prime;

                    ejk = this->baseMesh->verts.row(vj_index) - this->baseMesh->verts.row(vk_index);

					ejk_prime(0) = x(3 * vj_index) - x(3 * vk_index);
					ejk_prime(1) = x(3 * vj_index + 1) - x(3 * vk_index + 1);
					ejk_prime(2) = x(3 * vj_index + 2) - x(3 * vk_index + 2);

					Eigen::Vector3d tmp = ejk_prime - R[i] * dRij * Sj * ejk;
					third_term += cjk * tmp.dot(tmp);
				}
				second_term += cj_tilde * third_term;
			}
			// add energy
			energy += second_term;
		}

        return energy;
    }

};
