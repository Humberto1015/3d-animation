#include "angleSolver.h"

void AngleSolver::solve(const std::vector<double>& angles, const std::vector<int>& oriens, const std::vector<std::vector<int> >& neighbors){
    
    // ========================== get mesh edges ================================
    std::vector<std::pair<int, int> > edges;
    std::vector<std::vector<int> > visited(angles.size());
    for (int i = 0; i < visited.size(); ++i)
        visited[i] = std::vector<int>(angles.size(), 0);
    for (int i = 0; i < neighbors.size(); ++i){
        for (int j = 0; j < neighbors[i].size(); ++j){
            auto v_j = neighbors[i][j];
            if (i != v_j && (!visited[i][v_j] || !visited[v_j][i])){
                edges.emplace_back(std::make_pair(i, v_j));
                visited[i][v_j] = visited[v_j][i] = 1;
            }
        }
    }
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
    for (int i = 0; i < angles.size(); ++i){
        if (i == 0)
            x.add(IloNumVar(env, 0, 0, IloNumVar::Int));
        else
            x.add(IloNumVar(env, -1, 1, IloNumVar::Int));
    }
    // ==========================================================================

    // ========================== set objective =================================
    for (int i = 0; i < edges.size(); ++i){
        int s = edges[i].first;
        int t = edges[i].second;

        IloExpr term_i(env);
        IloExpr term_j(env);

        term_i += x[s] * 2 * M_PI + oriens[s] * angles[s];
        term_j += x[t] * 2 * M_PI + oriens[t] * angles[t];

        expr += IloPower((term_i - term_j), 2);

    }
    // ==========================================================================


    // ========================== solve the problem =============================
    model.add(IloMinimize(env, expr));
    IloCplex cplex(model);
    cplex.setParam(IloCplex::Param::OptimalityTarget, 1);

    if (!cplex.solve())
        std::cout << "[info] Failed to optimize the model\n";
    else
        std::cout << "[info] Solved.\n";
    // ==========================================================================

    IloNumArray res(env);
    cplex.getValues(res, x);

    this->solution.resize(angles.size());
    for (int i = 0; i < angles.size(); ++i)
        solution[i] = res[i];

    std::cout << "[info] Objective value = " << cplex.getObjValue() << "\n";
    env.end();
}