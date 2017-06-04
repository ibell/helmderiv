import CoolProp.CoolProp as CP, timeit

AS = CP.AbstractState('HEOS','n-Propane')
tau = 0.8
delta = 1.3
rhomolar = delta*AS.rhomolar_reducing()
T = AS.T_reducing()/tau

s = 0
N = 1000000
tic = timeit.default_timer()
for i in range(N):
    AS.specify_phase(CP.iphase_gas)
    AS.update(CP.DmolarT_INPUTS, rhomolar, T)
    s += AS.alphar() + AS.delta()*AS.dalphar_dDelta() + AS.tau()*AS.dalphar_dTau()
toc = timeit.default_timer()
print(s/N, (toc-tic)/N*1e6)