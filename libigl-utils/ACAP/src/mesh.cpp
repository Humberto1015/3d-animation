#include "mesh.h"

Mesh::Mesh(const Eigen::MatrixXd& verts, const Eigen::MatrixXi& faces){
    this->verts = verts;
    this->faces = faces;

    igl::adjacency_list(faces, this->neighbors, true);
    buildCotWeights();
}

void Mesh::show(){
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(this->verts, this->faces);
    viewer.launch();
}

void Mesh::buildCotWeights(){
    Eigen::SparseMatrix<double> C;
    igl::cotmatrix(this->verts, this->faces, C);

    int num_verts = verts.rows();

    this->cot_weights.resize(num_verts);

    for (int i = 0; i < num_verts; ++i){
        const int num_neighbors = this->neighbors[i].size();
        this->cot_weights[i].resize(num_neighbors);
        for (int j = 0; j < num_neighbors; ++j){
            int v_j = this->neighbors[i][j];
            double wij = C.coeff(i, v_j);
            const double eps = 1e-6f;
	        const double cotan_max = cos(eps) / sin(eps);
	        if (wij >= cotan_max) {
		        wij = cotan_max;
	        }
            this->cot_weights[i][j] = wij;
        }
    }
}