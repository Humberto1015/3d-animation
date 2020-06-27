#include <ilcplex/ilocplex.h>
#include <ilconcert/iloexpression.h>


class OrientationSolver{
private:

    std::vector<std::pair<int, int> > edges;
    std::vector<Eigen::Vector3d> axis;
    std::vector<double> coef;

    std::vector<int> initialSolution;

    double measureOrien(const Eigen::Vector3d& A, const Eigen::Vector3d& B){
        double eps_1 = 1e-6;
        double eps_2 = 1e-3;

        double inner_product = A.dot(B);

        if (fabs(inner_product) <= eps_1)
            return 1e-8;

        else if (inner_product > eps_1)
            return 1;

        else if (inner_product < -eps_1)
            return -1;
    }

    void setCoeff(){

        for (int i = 0; i < edges.size(); ++i){
            int s = edges[i].first;
            int t = edges[i].second;
            coef[i] = measureOrien(axis[s], axis[t]);
        }

        std::cout << "[info] Length of coefficient array = " << coef.size() << "\n";
    }

public:

    OrientationSolver(const std::vector<std::pair<int, int> >& _edges, const std::vector<Eigen::Vector3d>& _axis){
        edges = _edges;
        axis = _axis;
        coef.resize(edges.size());
        setCoeff();
    }

    void setInitSol(const std::vector<int>& initGuess){
        initialSolution.resize(initGuess.size());
        for (int i = 0; i < initialSolution.size(); ++i){
            initialSolution[i] = initGuess[i];
        }
    }


    void solve(std::vector<int>& solution){

        IloEnv env;
        IloModel model(env);
        IloNumVarArray x(env);
        IloExpr expr(env);
        IloNumVarArray startVar(env);
        IloNumArray startVal(env);

        // ========================== set variables =================================
        for (int i = 0; i < axis.size(); ++i){
            if (i == 0)
                x.add(IloNumVar(env, 1, 1, IloNumVar::Bool));
            else
                x.add(IloNumVar(env, 0, 1, IloNumVar::Bool));
        }
        // ===========================================================================

        // ========================== set objective ==================================
        for (int i = 0; i < edges.size(); ++i){
            int s = edges[i].first;
            int t = edges[i].second;
            // f(x) = 2 * x - 1, which maps 0 to -1, 1 to 1
            expr += (2 * x[s] - 1) * (2 * x[t] - 1) * coef[i];
        }
        // ===========================================================================

        model.add(IloMaximize(env, expr));

        IloCplex cplex(model);
        cplex.setParam(IloCplex::Param::OptimalityTarget, 3);

        // ========================== set initial guess ==============================
        for (int i = 0; i < axis.size(); ++i){
            startVar.add(x[i]);
            startVal.add(initialSolution[i]);
        }
        cplex.addMIPStart(startVar, startVal);
        startVal.end();
        startVar.end();

        std::cout << "hi\n";

        // ===========================================================================

        if (!cplex.solve()){
            std::cout << "[info] Failed to optimize the model\n";
        }
        else{
            std::cout << "[info] Solved.\n";
        }

        IloNumArray res(env);
        cplex.getValues(res, x);

        // receive solution
        solution.clear();
        solution.resize(axis.size());
        for (int i = 0; i < axis.size(); ++i)
            solution[i] = 2 * res[i] - 1;

        std::cout << "[info] Objective value = " << cplex.getObjValue() << "\n";
        env.end();
    }
};

class AngleSolver {
private:
    std::vector<std::pair<int, int> > edges;
    std::vector<double> angles;
    std::vector<int> orientation;

public:
    AngleSolver(const std::vector<std::pair<int, int> >& _edges, const std::vector<double>& _angles, const std::vector<int>& _orientation){
        std::cout << "[info] Initial the cplex environment.\n";
        edges = _edges;
        angles = _angles;
        orientation = _orientation;
    }

    void solve(std::vector<int>& solution){
        IloEnv env;
        IloModel model(env);
        IloNumVarArray x(env);
        IloExpr expr(env);

        // set variables
        size_t num_vars = 0;
        for (int i = 0; i < orientation.size(); ++i){
            if (i == 0)
                x.add(IloNumVar(env, 0, 0, IloNumVar::Int));
            else
                x.add(IloNumVar(env, -1, 1, IloNumVar::Int));
            num_vars++;
        }

        // set objective function
        for (auto e: edges){
            int s = e.first;
            int t = e.second;

            IloExpr term_i(env);
            IloExpr term_j(env);

            term_i += x[s] * 2 * igl::PI + orientation[s] * angles[s];
            term_j += x[t] * 2 * igl::PI + orientation[t] * angles[t];

            //expr += IloPower(((x[s] * 2 * igl::PI + orientation[s] * angles[s]) - (x[t] * 2 * igl::PI + orientation[t] * angles[t])), 2);
            expr += IloPower(term_i - term_j, 2);
        }

        model.add(IloMinimize(env, expr));
        IloCplex cplex(model);

        cplex.setParam(IloCplex::Param::OptimalityTarget, 1);
        cplex.setParam(IloCplex::Param::MIP::Limits::Solutions, 2);

        if (!cplex.solve()){
            std::cout << "[info] Failed to optimize the model\n";
        }
        else{
            std::cout << "[info] Solved.\n";
        }

        IloNumArray res(env);
        cplex.getValues(res, x);

        // receive solution
        solution.clear();
        for (int i = 0; i < num_vars; ++i)
            solution.emplace_back(res[i]);

        std::cout << "[info] Objective value = " << cplex.getObjValue() << "\n";

        env.end();
    }
};
