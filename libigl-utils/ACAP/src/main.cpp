#include "utils.h"

int main(int argc, char *argv[]){

    App app;
    // argument parser
    std::string option = argv[1];
    if (option == "--meshViewer"){
        std::string path = argv[2];
        app.testACAP(argv[2]);
    }

    else if (option == "--showAnimation"){
        app.showAnimation();
    }
    else if (option == "--testInterpolation"){
        std::string path_0 = argv[2];
        std::string path_1 = argv[3];
        app.testInterpolation(path_0, path_1);
    }
    else if (option == "--reconstruct"){
        std::string path = argv[2];
        app.reconstruct(path);
    }
    else if (option == "--genACAP"){
        std::string src = argv[2];
        std::string dst = argv[3];
        app.genACAP(src, dst);
    }

    return 0;
}
