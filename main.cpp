
#include "Helm.h"
#include "mixderiv.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <iomanip>

int main(){
    
    GenHelmDerivCoeffs coeff;
    coeff.d.resize(18); coeff.d << 4.0,1.0,1.0,2.0,2.0,1.0,3.0,6.0,6.0,2.0,3.0,1.0,1.0,1.0,2.0,2.0,4.0,1.0;
    coeff.n.resize(18); coeff.n << 0.042910051,1.7313671,-2.4516524,0.34157466,-0.46047898,-0.66847295,0.20889705,0.19421381,-0.22917851,-0.60405866,0.066680654,0.017534618,0.33874242,0.22228777,-0.23219062,-0.09220694,-0.47575718,-0.017486824;
    coeff.t.resize(18); coeff.t << 1,0.33,0.8,0.43,0.9,2.46,2.09,0.88,1.09,3.25,4.62,0.76,2.5,2.75,3.05,2.55,8.4,6.75;
    coeff.ld.resize(18); coeff.ld << 0.,0,0,0,0,1,1,1,1,2,2,0,0,0,0,0,0,0;
    coeff.ld_as_int.resize(18); coeff.ld_as_int << 0,0,0,0,0,1,1,1,1,2,2,0,0,0,0,0,0,0;
    coeff.cd.resize(18); coeff.cd << 0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0;
    coeff.lt.resize(18); coeff.lt << 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
    coeff.ct.resize(18); coeff.ct << 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;

    coeff.beta.resize(18); coeff.beta << 0,0,0,0,0,0,0,0,0,0,0,2.33,3.47,3.15,3.19,0.92,18.8,547.8;
    coeff.epsilon.resize(18); coeff.epsilon << 0,0,0,0,0,0,0,0,0,0,0,1.283,0.6936,0.788,0.473,0.8577,0.271,0.948;
    coeff.eta.resize(18); coeff.eta << 0,0,0,0,0,0,0,0,0,0,0,0.963,1.977,1.917,2.307,2.546,3.28,14.6;
    coeff.gamma.resize(18); coeff.gamma << 0,0,0,0,0,0,0,0,0,0,0,0.684,0.829,1.419,0.817,1.5,1.426,1.093;

    auto startTime = std::chrono::high_resolution_clock::now();
    GenHelmDeriv gen(coeff);
    
    Eigen::ArrayXd rhovec(0);
    DerivativeMatrices mat;
    GenHelmDerivDerivs d; // buffer
    mat.resize(1);
    
//    long N = 10;
    long N = 10000000;
    
    for (auto ld_int : {true}){
        for (auto only_a_couple : {false}){
            for (auto only_derivs : {true, false}){
                auto summer = 0.0;
                gen.all_ld_integers = ld_int;
                startTime = std::chrono::high_resolution_clock::now();
                for (auto i = 0; i < N; ++i){
                    if (only_a_couple){
                        gen.calc(0.8, 1.3, d, 1, 2);
                    }
                    else{
                        gen.calc(0.8, 1.3, d);
                    }
                    if (only_derivs){
                        summer += (  d.A(0,0)
                                   + d.A(0,1) + d.A(1,0)
                                   + d.A(0,2) + d.A(1,1) + d.A(2,0)
                                   + d.A(0,3) + d.A(1,2) + d.A(2,1) + d.A(3,0)
                                   );
                    }
                    else{
                        CubicNativeDerivProvider cube_nat(d, mat, rhovec);
                        MixDerivs mix(cube_nat);
                        summer += mix.d2psi_drhoidrhoj__constT(0,0);
                    }
                }
                auto elap = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count()/static_cast<double>(N)*1e6;
                std::cout << std::setprecision(20) << summer/N << " " << elap << std::endl;
            }
        }
    }
    return EXIT_SUCCESS;
}
