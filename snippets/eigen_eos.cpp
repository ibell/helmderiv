// C++ includes
#include <iostream>
#include <complex>
#include <chrono>
#include <iomanip>
using namespace std;

// autodiff include
#include <autodiff/forward.hpp>
#include <autodiff/reverse.hpp>
using namespace autodiff;

#ifndef M_LN2
#define M_LN2 0.693147180559945309417
#endif

const double one_over_M_LN2 = 1.0/M_LN2;

#include "Eigen/Dense"

// A type defining parameters for a function of interest
template <typename T>
struct Params
{
    T a;
    T b;
    T c;
};

template<typename T>
struct GenHelmDerivCoeffs{
    std::array<T,18> n,t,d,l,c,cd,ld,lt,ct,beta,epsilon,eta,gamma;
    std::array<int,18> ld_as_int;
};

template<typename T>
auto get_coeff(){
    GenHelmDerivCoeffs<T> coeff;
    coeff.d = {4.0,1.0,1.0,2.0,2.0,1.0,3.0,6.0,6.0,2.0,3.0,1.0,1.0,1.0,2.0,2.0,4.0,1.0};
    coeff.n = {0.042910051,1.7313671,-2.4516524,0.34157466,-0.46047898,-0.66847295,0.20889705,0.19421381,-0.22917851,-0.60405866,0.066680654,0.017534618,0.33874242,0.22228777,-0.23219062,-0.09220694,-0.47575718,-0.017486824};
    coeff.t = {1,0.33,0.8,0.43,0.9,2.46,2.09,0.88,1.09,3.25,4.62,0.76,2.5,2.75,3.05,2.55,8.4,6.75};
    coeff.ld = {0.,0,0,0,0,1,1,1,1,2,2,0,0,0,0,0,0,0};
    coeff.ld_as_int = {0,0,0,0,0,1,1,1,1,2,2,0,0,0,0,0,0,0};
    coeff.cd = {0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0};
    coeff.lt = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    coeff.ct = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    coeff.beta = {0,0,0,0,0,0,0,0,0,0,0,2.33,3.47,3.15,3.19,0.92,18.8,547.8};
    coeff.epsilon = {0,0,0,0,0,0,0,0,0,0,0,1.283,0.6936,0.788,0.473,0.8577,0.271,0.948};
    coeff.eta = {0,0,0,0,0,0,0,0,0,0,0,0.963,1.977,1.917,2.307,2.546,3.28,14.6};
    coeff.gamma = {0,0,0,0,0,0,0,0,0,0,0,0.684,0.829,1.419,0.817,1.5,1.426,1.093};
    return coeff;
}

template <typename T> auto sq(const T &x1){
    return x1*x1;
}

// The function that depends on parameters for which derivatives are needed
template <typename T, typename T2>
T alpha(T tau, T delta, const GenHelmDerivCoeffs<T2>& c)
{
    T o = 0, lntau = log(tau), lndelta = log(delta);
    for (auto i = 0; i < c.d.size(); ++i){
        o += c.n[i]*exp(c.d[i]*lndelta + c.t[i]*lntau + (-c.cd[i]*exp(c.ld[i]*log(delta)) - c.eta[i]*sq(delta-c.epsilon[i]) - c.beta[i]*sq(tau-c.gamma[i])));
    }
    return o;
}

int main()
{
    auto N = 100000;
    {
        typedef var vartype;
        vartype tau = 0.5+N*1e-8, delta = 0.7;  // the input variables

        auto coeff = get_coeff<double>();
        
        var dudtau, duddelta;
        auto startTime = std::chrono::high_resolution_clock::now();
    
        for (auto rr = 0; rr < N; ++rr){
            vartype a = alpha<vartype>(tau, delta, coeff);  // the output variable u
            DerivativesX dud = derivativesx(a);
            dudtau = dud(tau);
            duddelta = dud(delta);
        }
        auto elap = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count()/static_cast<double>(N)*1e6;
        std::cout << std::setprecision(20) << "REV: " << elap << " " << dudtau/N <<  std::endl;
    }
    {
        // Define a Nth order dual type using HigherOrderDual<N> construct.
        using vartype = dual;

        vartype tau = 0.5+N*1e-8, delta = 0.7;  // the input variables
        auto coeff = get_coeff<double>();
        double dudtau, duddelta;
        auto startTime = std::chrono::high_resolution_clock::now();
        auto N = 100000;
        for (auto rr = 0; rr < N; ++rr){
            dudtau = derivative(alpha<vartype, double>, wrt(tau), at(tau, delta, coeff));
            // duddelta = derivative(alpha<vartype>, wrt(delta, delta), at(tau, delta, coeff));
        }
        auto elap = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count()/static_cast<double>(N)*1e6;
        std::cout << std::setprecision(20) << "FWD: " << elap << " " << val(dudtau/N) << std::endl;
    }
    {
        auto startTime = std::chrono::high_resolution_clock::now();
        auto coeff_comdbl = get_coeff<std::complex<double>>();   
        double dudtau_csd = 0;
        for (auto rr = 0; rr < N; ++rr){
            std::complex<double> taustar(0.5+N*1e-8, 1e-100), deltastar(0.7, 0);
            dudtau_csd += (alpha<std::complex<double> >(taustar, deltastar, coeff_comdbl)/1e-100).imag();
        }
        auto elap = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count()/static_cast<double>(N)*1e6;
        std::cout << std::setprecision(20) << "CSD: " << elap << " " << dudtau_csd/N << std::endl;
    }
    {
        auto startTime = std::chrono::high_resolution_clock::now();
        auto coeff_double = get_coeff<double>();   
        double alphar_double = 0;
        for (auto rr = 0; rr < N; ++rr){
            alphar_double += alpha<double>(0.5+N*1e-8, 0.7, coeff_double);
        }
        auto elap = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count()/static_cast<double>(N)*1e6;
        std::cout << std::setprecision(20) << "FUN: " << elap << " " << alphar_double/N << std::endl;
    }

    // double v1 = autodiff::val(dudtau);
    // double v2 = dudtau_csd;
    // printf("%0.16e\n", v1-v2);
    // cout << "du/dtau = " << v1 << " " << v2 << endl;  // print the evaluated derivative du/dx
    // cout << "du/ddelta = " << duddelta << endl;  // print the evaluated derivative du/dx
}