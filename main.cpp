#include "Eigen/Dense"
#include <vector>
#include <iostream>

const std::vector<double> _d = {4,1,1,2,2,1,3,6,6,2,3,1,1,1,2,2,4,1};
const std::vector<double> _n = {0.042910051,1.7313671,-2.4516524,0.34157466,-0.46047898,-0.66847295,0.20889705,0.19421381,-0.22917851,-0.60405866,0.066680654,0.017534618,0.33874242,0.22228777,-0.23219062,-0.09220694,-0.47575718,-0.017486824};
const std::vector<double> _t = {1,0.33,0.8,0.43,0.9,2.46,2.09,0.88,1.09,3.25,4.62,0.76,2.5,2.75,3.05,2.55,8.4,6.75};
const std::vector<double> _l = {0,0,0,0,0,1,1,1,1,2,2,0,0,0,0,0,0,0};
const std::vector<double> _c = {0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0};

const std::vector<double> _beta = {0,0,0,0,0,0,0,0,0,0,0,2.33,3.47,3.15,3.19,0.92,18.8,547.8};
const std::vector<double> _epsilon = {0,0,0,0,0,0,0,0,0,0,0,1.283,0.6936,0.788,0.473,0.8577,0.271,0.948};
const std::vector<double> _eta = {0,0,0,0,0,0,0,0,0,0,0,0.963,1.977,1.917,2.307,2.546,3.28,14.6};
const std::vector<double> _gamma = {0,0,0,0,0,0,0,0,0,0,0,0.684,0.829,1.419,0.817,1.5,1.426,1.093};

int main(){
    Eigen::Map<const Eigen::VectorXd> n_w(&(_n[0]), _n.size()), t_w(&(_t[0]), _t.size()), d_w(&(_d[0]), _d.size());
    Eigen::Map<const Eigen::VectorXd> l_w(&(_l[0]), _l.size()), c_w(&(_c[0]), _c.size()) ;
    Eigen::Map<const Eigen::VectorXd> beta_w(&(_beta[0]), _beta.size()), epsilon_w(&(_epsilon[0]), _epsilon.size()),
                                      eta_w(&(_eta[0]), _eta.size()), gamma_w(&(_gamma[0]), _gamma.size());
    
    long N = 1000000; double summer = 0;
    auto startTime = std::chrono::system_clock::now();
    
    for (auto i = 0; i < N; ++i){
        const double tau = 0.8, delta = 1.3, log2tau = log2(tau), log2delta = log2(delta);
        Eigen::ArrayXd delta_minus_epsilon = delta-epsilon_w.array(), tau_minus_gamma = tau-gamma_w.array(), delta_to_l = Eigen::pow(delta, l_w.array());
        
        Eigen::VectorXd umat = -c_w.array()*delta_to_l - eta_w.array()*delta_minus_epsilon.square() - beta_w.array()*tau_minus_gamma.square();
        Eigen::VectorXd delta_du_ddelta = -c_w.array()*l_w.array()*delta_to_l-2*delta*eta_w.array()*delta_minus_epsilon;
        Eigen::VectorXd tau_du_dtau = -2*tau*beta_w.array()*tau_minus_gamma;
        
        Eigen::VectorXd armat = n_w.array()*Eigen::pow(2, (t_w*log2tau + d_w*log2delta + umat/M_LN2).array())*umat.array();
        
        double A00 = (armat).sum();
        double A01 = (armat.array()*(delta_du_ddelta.array()+d_w.array())).sum();
        double A10 = (armat.array()*(tau_du_dtau.array()+t_w.array())).sum();
        summer += A00 + A01 + A10;
    }
    auto elap = std::chrono::duration<double>(std::chrono::system_clock::now() - startTime).count()/static_cast<double>(N)*1e6;
    
    std::cout << summer/N << " " << elap << std::endl;
    return EXIT_SUCCESS;
}