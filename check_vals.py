import CoolProp.CoolProp as CP, timeit

AS = CP.AbstractState('HEOS','n-Propane')
tau = 0.8
delta = 1.3
rhomolar = delta*AS.rhomolar_reducing()
T = AS.T_reducing()/tau

s = 0
N = 10000
tic = timeit.default_timer()
for i in range(N):
    AS.specify_phase(CP.iphase_gas)
    AS.update(CP.DmolarT_INPUTS, rhomolar, T)
    # s += 0
    s += (AS.alphar() 
          + AS.delta()*AS.dalphar_dDelta() + AS.tau()*AS.dalphar_dTau() 
          + AS.tau()**2*AS.d2alphar_dTau2() + AS.delta()*AS.tau()*AS.d2alphar_dDelta_dTau() + AS.delta()**2*AS.d2alphar_dDelta2()
          + AS.delta()**3*AS.d3alphar_dDelta3() + AS.delta()*AS.tau()**2*AS.d3alphar_dDelta_dTau2() + AS.delta()**2*AS.tau()*AS.d3alphar_dDelta2_dTau() + AS.tau()**3*AS.d3alphar_dTau3()
          )

toc = timeit.default_timer()
print(s/N, (toc-tic)/N*1e6)