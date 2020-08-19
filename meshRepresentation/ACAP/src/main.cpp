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
        std::string setBlue = argv[4];
        app.reconstruct(path, setBlue);
    }
    else if (option == "--genACAP"){
        std::string src = argv[3];
        std::string dst = argv[4];
        app.genACAP(src, dst);
    }
    return 0;
}
