#include <ilcplex/ilocplex.h>
#include <ilconcert/iloexpression.h>
#include <Eigen/Geometry>
#include <iostream>

class AxisSolver{
private:
    void bfsInit();
    double getIndicator(const Eigen::Vector3d&, const Eigen::Vector3d&);

public:

    std::vector<int> solution;

    void solve(const std::vector<Eigen::Vector3d>&, const std::vector<std::vector<int> >&);
};