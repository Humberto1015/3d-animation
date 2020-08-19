#include <ilcplex/ilocplex.h>
#include <ilconcert/iloexpression.h>
#include <Eigen/Geometry>
#include <iostream>
#include <cmath>

class AngleSolver{
private:
    
public:

    std::vector<int> solution;

    void solve(const std::vector<double>&, const std::vector<int>&, const std::vector<std::vector<int> >&);

};