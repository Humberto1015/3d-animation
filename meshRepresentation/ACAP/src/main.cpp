#include "application.h"

int main(int argc, char *argv[]){

    std::string refPath = argv[1];

    App app(refPath);

    // the path of reference mesh
    std::string option = argv[2];

    // Load ACAP features in "ACAP-sequence", and render the animation
    if (option == "--animation"){
        app.animate();
    }

    // given an ACAP feature, reconstruct the mesh and visualize it
    else if (option == "--reconstruct"){
        std::string path = argv[3];
        app.reconstruct(path);
    }

    // mesh_location: the directory of meshes
    // target_ACAP_location: the target directory you want to save ACAP features
    else if (option == "--genACAP"){
        std::string mesh_location = argv[3];
        std::string target_ACAP_location = argv[4];
        app.genACAP(mesh_location, target_ACAP_location);
    }
    return 0;
}
