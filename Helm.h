

#include <cmath>

#include "Eigen/Dense"

#ifndef M_LN2
#define M_LN2 0.693147180559945309417
#endif

struct GenHelmDerivCoeffs{
    Eigen::ArrayXd n,t,d,l,c,cd,ld,lt,ct,beta,epsilon,eta,gamma;
};
struct GenHelmDerivDerivs{
    double A00,A10,A01,A20,A11,A02,A30,A21,A12,A03;
};

class GenHelmDeriv{

private:
    GenHelmDerivCoeffs coeffs;
    // Work arrays
    Eigen::ArrayXd tau_du_dtau, B1delta, delta_dB1delta_ddelta, B1tau, B2delta, B3delta, armat, umat, delta_minus_epsilon, tau_minus_gamma, delta_to_ld, tau_to_lt, delta_du_ddelta, delta2_d2u_ddelta2, delta3_d3u_ddelta3, delta2_d2B1delta_ddelta2, tau2_d2u_dtau2, tau3_d3u_dtau3, tau_dB1tau_dtau, B2tau, tau2_d2B1tau_dtau2, B3tau;

public:
    GenHelmDeriv(GenHelmDerivCoeffs &&coeffs) : coeffs(coeffs){};
    GenHelmDeriv(const GenHelmDerivCoeffs &coeffs) : coeffs(coeffs){};

    void calc(const double tau, const double delta, GenHelmDerivDerivs &derivs)
    {

        const double log2tau = log2(tau), log2delta = log2(delta);
        delta_minus_epsilon = delta- coeffs.epsilon, tau_minus_gamma = tau- coeffs.gamma, delta_to_ld = Eigen::pow(delta, coeffs.ld), tau_to_lt = Eigen::pow(tau, coeffs.lt);
    
        umat = -coeffs.cd*delta_to_ld - coeffs.ct*tau_to_lt - coeffs.eta*delta_minus_epsilon.square() - coeffs.beta*tau_minus_gamma.square();
    
        delta_du_ddelta = -coeffs.cd*coeffs.ld*delta_to_ld -2*delta*coeffs.eta*delta_minus_epsilon;
        delta2_d2u_ddelta2 = -coeffs.cd*coeffs.ld*(coeffs.ld-1)*delta_to_ld -2*(delta*delta)*coeffs.eta;
        delta3_d3u_ddelta3 = -coeffs.cd*coeffs.ld*(coeffs.ld-1)*(coeffs.ld-2)*delta_to_ld;
    
        tau_du_dtau = -coeffs.ct*coeffs.lt*tau_to_lt  -2*tau*coeffs.beta*tau_minus_gamma;
        tau2_d2u_dtau2 = -coeffs.ct*coeffs.lt*(coeffs.lt - 1)*tau_to_lt -2*(tau*tau)*coeffs.beta;
        tau3_d3u_dtau3 = -coeffs.ct*coeffs.lt*(coeffs.lt - 1)*(coeffs.lt- 2)*tau_to_lt;
    
        B1tau = tau_du_dtau + coeffs.t;
        // --
        tau_dB1tau_dtau = tau2_d2u_dtau2 + tau_du_dtau;
        B2tau = tau_dB1tau_dtau + B1tau.square() - B1tau;
        // --
        tau2_d2B1tau_dtau2 = tau3_d3u_dtau3 + 2*tau2_d2u_dtau2;
        B3tau = tau2_d2B1tau_dtau2 + 3*B1tau*tau_dB1tau_dtau - 2*tau_dB1tau_dtau + B1tau.cube() - 3*B1tau.square() + 2*B1tau;
    
        B1delta = delta_du_ddelta + coeffs.d;
        // --
        delta_dB1delta_ddelta = delta2_d2u_ddelta2 + delta_du_ddelta;
        B2delta = delta_dB1delta_ddelta + B1delta.square() - B1delta;
        // --
        delta2_d2B1delta_ddelta2 = delta3_d3u_ddelta3 + 2*delta2_d2u_ddelta2;
        B3delta = delta2_d2B1delta_ddelta2 + 3*B1delta*delta_dB1delta_ddelta - 2*delta_dB1delta_ddelta + B1delta.cube() - 3*B1delta.square() + 2*B1delta;
    
        armat = coeffs.n*Eigen::pow(2, (coeffs.t*log2tau + coeffs.d*log2delta + umat/M_LN2).array());
    
        derivs.A00 = armat.sum();
        derivs.A10 = (armat*B1tau).sum();
        derivs.A01 = (armat*B1delta).sum();
        derivs.A20 = (armat*B2tau).sum();
        derivs.A11 = (armat*B1tau*B1delta).sum();
        derivs.A02 = (armat*B2delta).sum();
        derivs.A30 = (armat*B3tau).sum();
        derivs.A21 = (armat*B2tau*B1delta).sum();
        derivs.A12 = (armat*B1tau*B2delta).sum();
        derivs.A03 = (armat*B3delta).sum();
    };
};