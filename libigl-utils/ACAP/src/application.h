#include "ACAP.h"
#include "npy.hpp"
#include <igl/readPLY.h>
#include <dirent.h>


class App{
private:
    ACAP* acap = nullptr;
    Eigen::MatrixXi meshFaces;
public:
    
    // Constructor
    // parameters: the path of reference mesh
    // the constructor initializes by setting the reference mesh
    App(const std::string&);

    // parameters: 
    // (1) ACAP feature vector
    // (2) target path you want to save to
    // this member function receives a ACAP feature and save it as numpy file
    void saveNumpy(const std::vector<double>&, const std::string&);

    // parameters: the path of numpy file which contains ACAP feature
    // this member function receives the path of a numpy file and
    // you will obtain an ACAP feature vector
    std::vector<double> loadNumpy(const std::string&);

    // parameters: the path of numpy file which contains ACAP feature
    // this member function renders the recovered mesh
    void reconstruct(const std::string& path);

    // to do: automatically count number of files under given directory
    // parameters:
    // (1) the directory which contains several meshes
    // (2) the target directory you want to put generated ACAP features
    // this member function converts meshes in a directory to ACAP features
    // and put them in the specified target directory
    void genACAP(const std::string&, const std::string&);
    
    // parameters: None
    // this member function loads ACAP feature from "../../../ACAP-sequence/"
    // and show the animation
    void animate();


    void debug(const std::string&);
    void debug(const std::string&, const std::string&);
};
