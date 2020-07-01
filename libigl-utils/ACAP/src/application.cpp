#include "application.h"

App::App(const std::string& refMeshPath){
    // get reference mesh
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::readPLY(refMeshPath, V, F);
    Mesh* mesh = new Mesh(V, F);
    this->meshFaces = F;
    this->acap = new ACAP(mesh, false);
}

void App::saveNumpy(const std::vector<double>& feat, const std::string& file_name){

    std::cout << "[info] Save to " + file_name << std::endl;
    std::vector<float> vals(feat.size());
    int idx = 0;

    for (int i = 0; i < feat.size(); ++i)
        vals[idx++] = (float)feat[i];

    const long unsigned shape[] = {(long unsigned)vals.size()};
    npy::SaveArrayAsNumpy(file_name, false, 1, shape, vals);
}

std::vector<double> App::loadNumpy(const std::string& path){

    std::vector<unsigned long> shape;
    bool fortran_order;
    std::vector<float> data;

    shape.clear();
    data.clear();
    npy::LoadArrayFromNumpy(path, shape, fortran_order, data);

    int idx = 0;
    std::vector<double> feat(shape[0]);
    for (int i = 0; i < shape[0]; ++i)
        feat[i] = (double)data[idx++];

    return feat;
}

void App::reconstruct(const std::string& path){
    
    auto f = loadNumpy(path);
    auto verts = acap->solveRecon(f);
    Mesh recon(verts, this->meshFaces);
    recon.show();
}

void App::genACAP(const std::string& dir_source, const std::string& dir_target){
    

    // compute & save ACAP feature for all models
    int num_samples = 10000;
    printf("[info] Start to generate %d ACAP files\n", num_samples);
    
    for (int i = 0; i < num_samples; ++i){
        // get the target mesh
        std::string path_target = dir_source + std::to_string(i) + ".ply";
        
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        igl::readPLY(path_target, V, F);

        // compute ACAP features
        printf("[info] Compute ACAP representation for %s...\n", path_target.c_str());
        Mesh* target_mesh = new Mesh(V, F);
        acap->setTargetMesh(target_mesh);
        acap->solveTransform();
        auto feat = acap->solveFeature();
        saveNumpy(feat, dir_target + std::to_string(i) + ".npy");
    }
}

void App::animate(){

    std::string seqDir = "../../../ACAP-sequence/";

    DIR *dir;
    struct dirent *ent;
    int num_files = -2; // . ..
    if ((dir = opendir(seqDir.c_str())) != NULL){
        while ((ent = readdir(dir)) != NULL){
            num_files++;
        }
    }

    std::vector<Eigen::MatrixXd> vertList(num_files);

    for (int i = 0; i < num_files; ++i){
        printf("[info] Processing frame %d\n", i);
        auto feat = loadNumpy(seqDir + std::to_string(i) + ".npy");

       

        auto verts = acap->solveRecon(feat);

        // bounding box normalization & rotate the mesh
        Eigen::Vector3d m = verts.colwise().minCoeff();
        Eigen::Vector3d M = verts.colwise().maxCoeff();
        Eigen::Vector3d center = (m + M) / 2;
        for (int j = 0; j < verts.rows(); ++j){

            double theta = igl::PI / 2;
            Eigen::Matrix3d m_x = Eigen::Matrix3d::Zero();
            Eigen::Matrix3d m_y = Eigen::Matrix3d::Zero();
            Eigen::Matrix3d m_z = Eigen::Matrix3d::Zero();
            m_x(2, 1) = 1;
            m_x(1, 2) = -1;
            m_y(0, 2) = 1;
            m_y(2, 0) = -1;
            m_z(0, 1) = -1;
            m_z(1, 0) = 1;
            m_x = theta * m_x;
            m_y = theta * m_y;
            m_z = theta * m_z;

            verts.row(j) -= center;
            verts.row(j) = m_z.exp() * m_y.exp() * m_z.exp() * verts.row(j).transpose();
        }

        vertList[i] = verts;
    }

    // render animation
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(vertList[0], this->meshFaces);

    viewer.core().is_animating = true;
    viewer.data().show_lines = false;
    viewer.core().animation_max_fps = 60;

    int count = 0;
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer& viewer){
        if (viewer.core().is_animating){
            viewer.data().clear();
            viewer.data().set_mesh(vertList[count], this->meshFaces);

            if ((++count) == vertList.size()){
                count = 0;
            }
        }
        return false;
    };
    viewer.launch();
}

void App::debug(const std::string& path){

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    igl::readPLY(path, V, F);
    Mesh* targetMesh = new Mesh(V, F);

    this->acap->setTargetMesh(targetMesh);
    this->acap->solveTransform();
    auto f = this->acap->solveFeature();

    auto verts = this->acap->solveRecon(f);
    Mesh recon(verts, this->meshFaces);
    recon.show();
}

void App::debug(const std::string& path_0, const std::string& path_1){

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    igl::readPLY(path_0, V, F);
    Mesh* targetMesh = new Mesh(V, F);

    this->acap->setTargetMesh(targetMesh);
    this->acap->solveTransform();
    auto f_0 = this->acap->solveFeature();
    
    igl::readPLY(path_1, V, F);
    targetMesh = new Mesh(V, F);
    this->acap->setTargetMesh(targetMesh);
    this->acap->solveTransform();
    auto f_1 = this->acap->solveFeature();

    std::vector<double> f_mid(f_0.size());

    double t = 0.5;
    for (int i = 0; i < f_mid.size(); ++i)
        f_mid[i] = f_0[i] + t * (f_1[i] - f_0[i]);


    auto verts = this->acap->solveRecon(f_mid);
    Mesh recon(verts, this->meshFaces);
    recon.show();

}
