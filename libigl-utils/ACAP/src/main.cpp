#include "application.h"

int main(int argc, char *argv[]){

    std::string refPath = argv[1];

    App app(refPath);

    // argument parser
    std::string option = argv[2];

    if (option == "--animation"){
        app.animate();
    }
    else if (option == "--reconstruct"){
        std::string path = argv[3];
        app.reconstruct(path);
    }
    else if (option == "--genACAP"){
        std::string src = argv[3];
        std::string dst = argv[4];
        app.genACAP(src, dst);
    }
    else if (option == "--debug"){
        std::string path_0 = argv[3];
        std::string path_1 = argv[4];
        app.debug(path_0, path_1);
        //app.debug(path_0);
    }

    return 0;
}
