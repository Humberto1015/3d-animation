#include <igl/readPLY.h>
#include "rimd.h"

#include <string>

void save_as_binary_file(std::vector<std::vector<Eigen::Matrix3d> >& rimd_vectors, std::string file_name){

    std::cout << "Save to " + file_name << std::endl;

    std::ofstream fp;
    fp.open(file_name, std::ios::out| std::ios::binary);
	for (int i = 0; i < rimd_vectors.size(); ++i) {

        auto one_ring = rimd_vectors[i];

        for (int j = 0; j < one_ring.size(); ++j){
            for (int k = 0; k < 3; ++k){
                for (int l = 0; l < 3; ++l){
                    float value = one_ring[j](k, l);
                    fp.write(reinterpret_cast<char*>(&value), sizeof(value));
                }
            }
        }
	}
	fp.close();
}

std::vector<std::vector<Eigen::Matrix3d> > get_RIMD_from_file(std::string header, std::string feature) {

	// Read the header file to obtain neighbor number
	std::ifstream fin(header, std::ios::in | std::ios::binary);
	int buffer[10240];
	fin.read((char*)buffer, sizeof(buffer));
	fin.close();

	int num_vertices = buffer[0];
	std::vector<int> num_neighbors(num_vertices);
	for (int i = 0; i < num_vertices; i++) {
		num_neighbors[i] = (int)buffer[i + 1];
	}


	std::ifstream in(feature, std::ios::in | std::ios::binary);
	std::vector<std::vector<Eigen::Matrix3d>> RIMD_vectors;

	std::vector<float> values;
	float value;
	while (in.read(reinterpret_cast<char*>(&value), sizeof(float))) {
		values.push_back(value);
	}

	// build RIMD vectors
	int index = 0;
	for (int i = 0; i < num_neighbors.size(); i++) {
		std::vector<Eigen::Matrix3d> elements;

		int nb_num = num_neighbors[i];
		// collect R matrices
		for (int j = 0; j < nb_num; j++) {
			Eigen::Matrix3d R;
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					R(k, l) = (double)values[index++];
				}
			}
			elements.push_back(R);
		}
		// collect S matrix
		Eigen::Matrix3d S;
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				S(j, k) = (double)values[index++];
			}
		}
		elements.push_back(S);

		RIMD_vectors.push_back(elements);
	}

	return RIMD_vectors;
}

int main(int argc, char *argv[]){

    // the directory path of mesh files
    std::string dir_source = argv[1];
    // the directory path of generated RIMD files
    std::string dir_target = argv[2];

    // get the base mesh
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    std::string name = "0.ply";
    std::string path_base = dir_source + name;
    igl::readPLY(path_base, V, F);
    Mesh base_mesh(V, F);

    RIMD rimd(base_mesh, base_mesh);
    /*
    igl::readPLY(dir_source + "2.ply", V, F);
    Mesh target_mesh(V, F);

    RIMD rimd(base_mesh, target_mesh);
    auto feat = rimd.features;
    auto verts = rimd.solveReconstruct(feat);
    Mesh recon(verts, F);
    recon.show();
    */

    for (int i = 0; i < 10; ++i){
        std::string header_path = "../../rimd-data/Animal_all/test/header.b";
        std::string feat_path = "../../" + std::to_string(i) + ".b";
        auto data = get_RIMD_from_file(header_path, feat_path);
        std::cout << data.size() << std::endl;
        auto verts = rimd.solveReconstruct(data);
        Mesh target(verts, F);
        target.show();
    }

    /*
    //A header file containing the neighbor number of each vertex
    std::cout << "Building the header file..." << std::endl;
	std::ofstream fp;
    std::string header_path = dir_target + "header.b";
	fp.open(header_path);
	int num_verts = base_mesh.verts.rows();
	fp.write(reinterpret_cast<char*>(&num_verts), sizeof(num_verts));
	for (int i = 0; i < num_verts; ++i) {
		int num_neighbors = base_mesh.neighbors[i].size();
		fp.write(reinterpret_cast<char*>(&num_neighbors), sizeof(num_neighbors));
	}
	fp.close();

    // content file containing rimd information of the mesh
    int num_samples = 100;
    for (int i = 0; i < num_samples; ++i){
        // get the target mesh
        std::string path_target = dir_source + std::to_string(i) + ".ply";
        igl::readPLY(path_target, V, F);
        Mesh target_mesh(V, F);
        // compute rimd features
        printf("Compute RIMD features for %s...\n", path_target.c_str());
        RIMD rimd(base_mesh, target_mesh);
        auto rimd_features = rimd.features;
        printf("Done.\n");
        // save as files
        save_as_binary_file(rimd_features, dir_target + std::to_string(i) + ".b");
    }
    */

    return 0;
}
