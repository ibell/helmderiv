//
//  mixderiv.h
//  HelmDeriv
//
//  Created by Ian on 9/30/17.
//

#ifndef mixderiv_h
#define mixderiv_h

#include "helm.h"

inline bool Kronecker(std::size_t i, std::size_t j){ return i == j; }
inline double POW2(double x){ return x*x; }
inline double POW4(double x){ return POW2(x)*POW2(x); }

// This class defines the interface that must be provided by the implementer.  This could be
// calculations coming from a cubic EOS, CoolProp's HEOS backend, REFPROP, etc.
//
// In general, the intention is that the class is immutable and does not do any calculations itself;
// all derivatives should be cached internally
class AbstractNativeDerivProvider{
private:
    const Eigen::ArrayXd &m_rhovec;
public:
    AbstractNativeDerivProvider(const Eigen::ArrayXd &rhovec) : m_rhovec(rhovec) {};
    const Eigen::ArrayXd &rhovec() const{ return m_rhovec; };
    virtual double tau() const = 0;
    virtual double delta() const = 0;
    virtual double A(std::size_t i, std::size_t j) const = 0;
    virtual double Tr() const = 0;
    virtual double dTr_drhoi(std::size_t i) const = 0;
    virtual double dTr_dxi__constxj(std::size_t i) const = 0;
    virtual double d2Tr_drhoidrhoj(std::size_t i, std::size_t j) const = 0;
    virtual double d2Tr_dxidxj(std::size_t i, std::size_t j) const = 0;
    virtual double rhor() const = 0;
    virtual double drhor_drhoi(std::size_t i) const = 0;
    virtual double drhor_dxi__constxj(std::size_t i) const =  0; 
    virtual double d2rhor_drhoidrhoj(std::size_t i, std::size_t j) const = 0;
    virtual double d2rhor_dxidxj(std::size_t i, std::size_t j) const = 0;
    virtual double dalpha_dxi__taudeltaxj(std::size_t i) const = 0;
    virtual double d2alpha_dxi_ddelta__consttauxj(std::size_t i) const = 0;
    virtual double d2alpha_dxi_dtau__constdeltaxj(std::size_t i) const = 0;
    virtual double d2alpha_dxidxj__consttaudelta(std::size_t i, std::size_t j) const = 0;
};

class DerivativeMatrices{
public:
    Eigen::ArrayXd dalpha_dxi__taudeltaxj, d2alpha_dxi_ddelta__consttauxj, d2alpha_dxi_dtau__constdeltaxj;
    Eigen::MatrixXd d2alpha_dxidxj__consttaudelta;
    void resize(std::size_t N){
        std::function<void(Eigen::ArrayXd &A)> init = [N](Eigen::ArrayXd &A){ A.resize(N); A.fill(0.0);};
        std::function<void(Eigen::MatrixXd &A)> initmat = [N](Eigen::MatrixXd &A){ A.resize(N,N); A.fill(0.0);};
        init(dalpha_dxi__taudeltaxj);
        init(d2alpha_dxi_ddelta__consttauxj);
        init(d2alpha_dxi_dtau__constdeltaxj);
        initmat(d2alpha_dxidxj__consttaudelta);
    }
};

class CubicNativeDerivProvider : public AbstractNativeDerivProvider{
private:
    const GenHelmDerivDerivs &m_vals;
    const DerivativeMatrices &m_mats;
public:
    CubicNativeDerivProvider(const GenHelmDerivDerivs & vals, const DerivativeMatrices & mats, const Eigen::ArrayXd &rhovec) : AbstractNativeDerivProvider(rhovec), m_vals(vals), m_mats(mats){};
    virtual double tau() const override{return m_vals.tau;};
    virtual double delta() const  override{return m_vals.delta;};
    virtual double A(std::size_t itau, std::size_t idelta) const override{ return m_vals.A(itau,idelta); };
    virtual double Tr() const override{ return 1.0; };
    virtual double dTr_drhoi(std::size_t i) const override{ return 0.0; };
    virtual double dTr_dxi__constxj(std::size_t i) const override{ return 0.0; };
    virtual double d2Tr_drhoidrhoj(std::size_t i, std::size_t j) const override{ return 0.0; };
    virtual double d2Tr_dxidxj(std::size_t i, std::size_t j) const override{ return 0.0; };
    virtual double rhor() const override{ return 1.0; };
    virtual double drhor_drhoi(std::size_t i) const override{ return 0.0; };
    virtual double drhor_dxi__constxj(std::size_t i) const override { return 0.0; };
    virtual double d2rhor_drhoidrhoj(std::size_t i, std::size_t j) const override{ return 0.0; };
    virtual double d2rhor_dxidxj(std::size_t i, std::size_t j) const override{ return 0.0; };
    virtual double dalpha_dxi__taudeltaxj(std::size_t i) const override{ return m_mats.dalpha_dxi__taudeltaxj(i);};
    virtual double d2alpha_dxi_ddelta__consttauxj(std::size_t i) const override{ return m_mats.d2alpha_dxi_ddelta__consttauxj(i);};
    virtual double d2alpha_dxi_dtau__constdeltaxj(std::size_t i) const override{ return m_mats.d2alpha_dxi_dtau__constdeltaxj(i);};
    virtual double d2alpha_dxidxj__consttaudelta(std::size_t i, std::size_t j) const override{return m_mats.d2alpha_dxidxj__consttaudelta(i,j);};
};

class MixDerivs{
public:
    const AbstractNativeDerivProvider &m_ders;
    const Eigen::ArrayXd &m_rhovec;
    const double R;
    const double delta, tau, rhor, Tr, T, rho, rhorTr;
    MixDerivs(const AbstractNativeDerivProvider &ders)
    : m_ders(ders), m_rhovec(m_ders.rhovec()), R(8.3144598), delta(m_ders.delta()), tau(m_ders.tau()), rhor(m_ders.rhor()), Tr(m_ders.Tr()), T(Tr/tau), rho(rhor*delta), rhorTr(rhor*Tr)
    {};
    double ddelta_drhoi__constTrhoj(std::size_t i) const{
        return (rhor - rho*m_ders.drhor_drhoi(i))/(rhor*rhor);
    }
    double d2delta_drhoidrhoj__constT(std::size_t i, std::size_t j) const{
        return (POW2(rhor)*(m_ders.drhor_drhoi(j)-m_ders.drhor_drhoi(i)-rho*m_ders.d2rhor_drhoidrhoj(i,j))-(rhor-rho*m_ders.drhor_drhoi(i))*(2*rhor*m_ders.drhor_drhoi(j)))/POW4(rhor);
    }
    double dtau_drhoi__constTrhoj(std::size_t i) const{
        const double one_over_T = tau/m_ders.Tr();
        return m_ders.dTr_drhoi(i)*one_over_T;
    }
    double d2tau_drhoidrhoj__constT(std::size_t i, std::size_t j) const{
        const double one_over_T = tau/m_ders.Tr();
        return m_ders.d2Tr_drhoidrhoj(i,j)*one_over_T;
    }
    double dxj_drhoi__constTrhoj(std::size_t j, std::size_t i) const{
        return (rho*Kronecker(i,j)-m_rhovec(i))/(rho*rho);
    }
    double d2xk_drhoidrhoj__constT(std::size_t k, std::size_t i, std::size_t j) const{
        return (rho*(Kronecker(i,k)-Kronecker(j,k)) - 2*(rho*Kronecker(i,k)-m_rhovec(k)))/(rho*rho*rho);
    }
    double drhorTr_dxi__xj(std::size_t i) const{
        return m_ders.Tr()*m_ders.drhor_dxi__constxj(i) + m_ders.rhor()*m_ders.dTr_dxi__constxj(i);
    }
    double d2rhorTr_dxidxj__consttaudelta(std::size_t i, std::size_t j) const{
        return (m_ders.Tr()*m_ders.d2rhor_dxidxj(i,j)
                + m_ders.dTr_dxi__constxj(j)*m_ders.drhor_dxi__constxj(i)
                + m_ders.rhor()*m_ders.d2Tr_dxidxj(i,j)
                + m_ders.drhor_dxi__constxj(j)*m_ders.dTr_dxi__constxj(i)
                );
    }
    double dpsi_dxi__consttaudeltaxj(std::size_t i) const{
        return delta*R/tau*(A(0,0)*drhorTr_dxi__xj(i) + rhorTr*m_ders.dalpha_dxi__taudeltaxj(i));
    }
    double d2psi_dxi_ddelta__consttauxj(std::size_t i) const{
        return R/tau*(drhorTr_dxi__xj(i)*(A(0,1)+A(0,0))+rhorTr*(delta*m_ders.d2alpha_dxi_ddelta__consttauxj(i) + m_ders.dalpha_dxi__taudeltaxj(i)));
    }
    double d2psi_dxi_dtau__constdeltaxj(std::size_t i) const{
        return delta*R/POW2(tau)*(drhorTr_dxi__xj(i)*(A(1,0)-A(0,0))+rhorTr*(tau*m_ders.d2alpha_dxi_dtau__constdeltaxj(i) - m_ders.dalpha_dxi__taudeltaxj(i)));
    }
    double d2psi_dxidxj__consttaudelta(std::size_t i, std::size_t j) const{
        return delta*R/tau*(A(0,0)*d2rhorTr_dxidxj__consttaudelta(i,j)
                            + m_ders.dalpha_dxi__taudeltaxj(i)*drhorTr_dxi__xj(j)
                            + rhorTr*m_ders.d2alpha_dxidxj__consttaudelta(i,j)
                            + m_ders.dalpha_dxi__taudeltaxj(j)*drhorTr_dxi__xj(i)
                            );
    }
    /** Combined ideal-gas and residual contributions into A
     */
    inline double A(std::size_t itau, std::size_t idelta) const{
        return m_ders.A(itau, idelta);
    };
    double dpsi_ddelta() const{
        return (m_ders.rhor()*R*T)*(A(0,1) + A(0,0));
    };
    double dpsi_dtau() const{
        return (m_ders.rhor()*delta*R*Tr/(tau*tau))*(A(1,0) - A(0,0));
    }
    double d2psi_ddelta2() const {
        return (m_ders.rhor()*R*T/delta)*(A(0,2) + 2*A(0,1));
    }
    double d2psi_ddelta_dtau() const {
        return (m_ders.rhor()*R*Tr/(tau*tau))*(A(1,0) - A(0,0) - A(0,1) + A(1,1));
    }
    double d2psi_dtau2() const{
        return (m_ders.rhor()*delta*R*Tr/(tau*tau*tau))*(A(2,0) - 2*A(1,0) + 2*A(0,0));
    }
    double d_dpsi_ddelta_drhoi__constrhoj(std::size_t i) const{
        double summer = 0;
        auto N = m_rhovec.size();
        for (auto m = 0; m < N; ++m){
            summer += d2psi_dxi_ddelta__consttauxj(m)*dxj_drhoi__constTrhoj(m,i);
        }
        return d2psi_ddelta2()*ddelta_drhoi__constTrhoj(i) + d2psi_ddelta_dtau()*dtau_drhoi__constTrhoj(i) + summer;
    }
    double d_dpsi_dtau_drhoi__constrhoj(std::size_t i) const{
        double summer = 0;
        auto N = m_rhovec.size();
        for (auto m = 0; m < N; ++m){
            summer += d2psi_dxi_dtau__constdeltaxj(m)*dxj_drhoi__constTrhoj(m,i);
        }
        return d2psi_ddelta_dtau()*ddelta_drhoi__constTrhoj(i) + d2psi_dtau2()*dtau_drhoi__constTrhoj(i) + summer;
    }
    double d_dpsi_dxm_drhoi__constTrhoi(std::size_t m, std::size_t i) const{
        double summer = 0;
        auto N = m_rhovec.size();
        for (auto n = 0; n < N; ++n){
            summer += d2psi_dxidxj__consttaudelta(m,n)*dxj_drhoi__constTrhoj(n,i);
        }
        return (d2psi_dxi_ddelta__consttauxj(m)*ddelta_drhoi__constTrhoj(i)
                + d2psi_dxi_dtau__constdeltaxj(m)*dtau_drhoi__constTrhoj(i) 
                + summer);
    }
    double dpsi_drhoi__constTrhoj(std::size_t i) const{
        double summer = 0;
        auto N = m_rhovec.size();
        for (auto m = 0; m < N; ++m){
            summer += dpsi_dxi__consttaudeltaxj(m)*dxj_drhoi__constTrhoj(m,i);
        }
        return (dpsi_ddelta()*ddelta_drhoi__constTrhoj(i)
                + dpsi_dtau()*dtau_drhoi__constTrhoj(i)
                + summer);
    }
    double d2psi_drhoidrhoj__constT(std::size_t i, std::size_t j) const{
        double summer = 0;
        auto N = m_rhovec.size();
        for (auto m = 0; m < N; ++m){
            summer += (dpsi_dxi__consttaudeltaxj(m)*d2xk_drhoidrhoj__constT(m,i,j)
                       + d_dpsi_dxm_drhoi__constTrhoi(m,j)*dxj_drhoi__constTrhoj(m, i));
        }
        return (dpsi_ddelta()*d2delta_drhoidrhoj__constT(i,j)
                + d_dpsi_ddelta_drhoi__constrhoj(i)*ddelta_drhoi__constTrhoj(i)
                + dpsi_dtau()*d2tau_drhoidrhoj__constT(i,j)
                + d_dpsi_dtau_drhoi__constrhoj(i)*dtau_drhoi__constTrhoj(i)
                + summer);
    }
};

#endif /* mixderiv_h */
