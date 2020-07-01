#include "axisSolver.h"

void AxisSolver::bfsInit(){}
double AxisSolver::getIndicator(const Eigen::Vector3d& A, const Eigen::Vector3d& B){

    double eps = 1e-6;
    double inner_product = A.dot(B);

    if (fabs(inner_product) <= eps)
        return 1e-8;

    else if (inner_product > eps)
        return 1;

    else if (inner_product < -eps)
        return -1;
}

void AxisSolver::solve(const std::vector<Eigen::Vector3d>& axis, const std::vector<std::vector<int> >& neighbors){

    

    // ========================== get mesh edges ================================
    std::vector<std::pair<int, int> > edges;
    std::vector<std::vector<int> > visited(axis.size());
    for (int i = 0; i < visited.size(); ++i)
        visited[i] = std::vector<int>(axis.size(), 0);
    for (int i = 0; i < neighbors.size(); ++i){
        for (int j = 0; j < neighbors[i].size(); ++j){
            auto v_j = neighbors[i][j];
            if (i != v_j && (!visited[i][v_j] || !visited[v_j][i])){
                edges.emplace_back(std::make_pair(i, v_j));
                visited[i][v_j] = visited[v_j][i] = 1;
            }
        }
    }
    //delete visited;
    // ==========================================================================


    // ========================== set cplex config ==============================
    IloEnv env;
    IloModel model(env);
    IloNumVarArray x(env);
    IloExpr expr(env);
    IloNumVarArray startVar(env);
    IloNumArray startVal(env);
    // ==========================================================================


    // ========================== set variables =================================
    for (int i = 0; i < axis.size(); ++i){
        if (i == 0)
            x.add(IloNumVar(env, 1, 1, IloNumVar::Bool));
        else
            x.add(IloNumVar(env, 0, 1, IloNumVar::Bool));
    }
    // ==========================================================================


    // ========================== set coefficients ==============================
    std::vector<double> coefs(edges.size());
    for (int i = 0; i < edges.size(); ++i){
        int s = edges[i].first;
        int t = edges[i].second;
        coefs[i] = getIndicator(axis[s], axis[t]);
    }
    // ==========================================================================


    // ========================== set objective =================================
    for (int i = 0; i < edges.size(); ++i){
        int s = edges[i].first;
        int t = edges[i].second;
        // f(x) = 2 * x - 1, which maps 0 to -1, 1 to 1
        expr += (2 * x[s] - 1) * (2 * x[t] - 1) * coefs[i];
    }
    // ==========================================================================


    // ========================== solve the problem =============================
    model.add(IloMaximize(env, expr));
    IloCplex cplex(model);
    cplex.setParam(IloCplex::Param::OptimalityTarget, 3);

    if (!cplex.solve())
        std::cout << "[info] Failed to optimize the model\n";
    else
        std::cout << "[info] Solved.\n";
    // ==========================================================================

    IloNumArray res(env);
    cplex.getValues(res, x);

    this->solution.resize(axis.size());
    for (int i = 0; i < axis.size(); ++i)
        solution[i] = 2 * res[i] - 1;

    std::cout << "[info] Objective value = " << cplex.getObjValue() << "\n";
    env.end();
}
