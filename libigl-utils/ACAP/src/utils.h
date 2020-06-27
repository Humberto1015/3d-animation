#include "ACAP.h"
#include "npy.hpp"
#include <igl/readPLY.h>
#include <dirent.h>

class App{
public:

    void saveAsNumpy(const std::vector<double> feat, const std::string file_name){

        std::cout << "[info] Save to " + file_name << std::endl;
        std::vector<float> vals(feat.size());
        int idx = 0;

        for (int i = 0; i < feat.size(); ++i)
            vals[idx++] = (float)feat[i];

        const long unsigned shape[] = {(long unsigned)vals.size()};
        npy::SaveArrayAsNumpy(file_name, false, 1, shape, vals);
    }

    std::vector<double> loadACAP(const std::string path){

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

    void reconstruct(const std::string path){
        // set base mesh
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        igl::readPLY("../../../mesh-data/cats/0.ply", V, F);
        Mesh baseMesh(V, F);

        ACAP acap(baseMesh);
        auto f = loadACAP(path);
        auto verts = acap.solveRecon(f);
        Mesh recon(verts, F);
        recon.show();
    }

    // given a mesh
    // 1. convert it to ACAP feature
    // 2. recover it to mesh
    void testACAP(const std::string path){

        // set base mesh
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        igl::readPLY("../../../mesh-data/SMPL/0.ply", V, F);
        Mesh baseMesh(V, F);

        ACAP acap(baseMesh);

        igl::readPLY(path, V, F);
        Mesh targetMesh(V, F);

        acap.setTargetMesh(targetMesh);
        acap.solveTransform();
        auto f = acap.solveFeature();
        auto verts = acap.solveRecon(f);
        Mesh recon(verts, F);
        recon.show();

    }

    void testInterpolation(const std::string path_0, const std::string path_1){
        // set base mesh
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        igl::readPLY("../../../mesh-data/SMPL/0.ply", V, F);
        Mesh baseMesh(V, F);
        ACAP acap(baseMesh);

        igl::readPLY(path_0, V, F);
        Mesh mesh_0(V, F);

        igl::readPLY(path_1, V, F);
        Mesh mesh_1(V, F);


        acap.setTargetMesh(mesh_0);
        acap.solveTransform();
        auto f_0 = acap.solveFeature();

        acap.setTargetMesh(mesh_1);
        acap.solveTransform();
        auto f_1 = acap.solveFeature();

        std::vector<double> f_mid(f_0.size());
        for (int i = 0; i < f_mid.size(); ++i){
            f_mid[i] = 0.5 * (f_0[i] + f_1[i]);
        }

        /*
        for (int i = 0; i < 6890; ++i){

            std::vector<double> coef_0 = {f_0[10 * i + 0], f_0[10 * i + 1], f_0[10 * i + 2], f_0[10 * i + 3]};
            std::vector<double> coef_1 = {f_1[10 * i + 0], f_1[10 * i + 1], f_1[10 * i + 2], f_1[10 * i + 3]};
            Eigen::Quaterniond Q_0(coef_0[0], coef_0[1], coef_0[2], coef_0[3]);
            Eigen::Quaterniond Q_1(coef_1[0], coef_1[1], coef_1[2], coef_1[3]);
            auto Q_mid = Q_0.slerp(0.5, Q_1);

            f_mid[i * 10 + 0] = Q_mid.w();
            f_mid[i * 10 + 1] = Q_mid.x();
            f_mid[i * 10 + 2] = Q_mid.y();
            f_mid[i * 10 + 3] = Q_mid.z();

            f_mid[i * 10 + 4] = 0.5 * (f_0[10 * i + 4] + f_1[10 * i + 4]);
            f_mid[i * 10 + 5] = 0.5 * (f_0[10 * i + 5] + f_1[10 * i + 5]);
            f_mid[i * 10 + 6] = 0.5 * (f_0[10 * i + 6] + f_1[10 * i + 6]);
            f_mid[i * 10 + 7] = 0.5 * (f_0[10 * i + 7] + f_1[10 * i + 7]);
            f_mid[i * 10 + 8] = 0.5 * (f_0[10 * i + 8] + f_1[10 * i + 8]);
            f_mid[i * 10 + 9] = 0.5 * (f_0[10 * i + 9] + f_1[10 * i + 9]);
        }
        */

        auto verts = acap.solveRecon(f_mid);
        Mesh recon(verts, F);
        recon.show();
    }

    void genACAP(const std::string dir_source, const std::string dir_target){
        // step 1. set the base mesh
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        std::string name = "0.ply";
        std::string path_base = dir_source + name;
        igl::readPLY(path_base, V, F);
        Mesh base_mesh(V, F);
        ACAP acap(base_mesh);

        // step 2. compute & save ACAP feature for all models
        int num_samples = 10000;

        printf("[info] Start to generate %d ACAP files\n", num_samples);

        for (int i = 0; i < num_samples; ++i){
            // get the target mesh
            std::string path_target = dir_source + std::to_string(i) + ".ply";
            igl::readPLY(path_target, V, F);
            Mesh target_mesh(V, F);
            // compute ACAP features
            printf("[info] Compute ACAP representation for %s...\n", path_target.c_str());
            acap.setTargetMesh(target_mesh);
            acap.solveTransform();
            auto feat = acap.solveFeature();
            printf("[info] Done.\n");

            saveAsNumpy(feat, dir_target + std::to_string(i) + ".npy");
        }
    }

    void showAnimation(){

        std::string seqDir = "../../../ACAP-sequence/";

        DIR *dir;
        struct dirent *ent;
        int num_files = -2; // . ..
        if ((dir = opendir(seqDir.c_str())) != NULL){
            while ((ent = readdir(dir)) != NULL){
                num_files++;
            }
        }

        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        igl::readPLY("../../../mesh-data/SMPL/0.ply", V, F);
        Mesh base(V, F);
        ACAP acap(base);

        std::vector<Eigen::MatrixXd> vertList;

        for (int i = 0; i < num_files; ++i){
            printf("[info] Processing frame %d\n", i);
            auto feat = loadACAP(seqDir + std::to_string(i) + ".npy");
            auto verts = acap.solveRecon(feat);

            // bounding box normalization
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

            vertList.emplace_back(verts);
            //vertList.emplace_back(verts); // add an extrac frame to slow down the animation
        }

        // show animation
        igl::opengl::glfw::Viewer viewer;
        viewer.data().set_mesh(vertList[0], F);

        viewer.core().is_animating = true;
        viewer.data().show_lines = false;
        viewer.core().animation_max_fps = 60;

        int count = 0;
        viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer& viewer){
            if (viewer.core().is_animating){
                viewer.data().clear();
                viewer.data().set_mesh(vertList[count], F);

                if ((++count) == vertList.size()){
                    count = 0;
                }
            }

            return false;
        };
        viewer.launch();

    }



};
