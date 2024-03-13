'''Symbolically work out perturbation theory
for rotational density matrix evolution.
Only for a specific size matrix.
Proof by induction requires some manual intervention.'''

import sympy as sy

# polarizability orthogonal to diatomic axis
aperp = sy.symbols('aperp')
# difference polarizability (a_parallel - a_perp)
da = sy.symbols('da')
# Q = <jm|cos^2(theta)|j'm'>
Qd = sy.symbols('Qd0:5') # diagonal
Qu = sy.symbols('Qu0:5') # 2 above diagonal
Ql = sy.symbols('Ql0:5') # 2 below diagonal
eps0 = sy.symbols('eps0')
E = sy.symbols('E')
t,dt = sy.symbols('t dt')
hbar = sy.symbols('hbar')
T,W,Q,wE = sy.symbols('T W Q wE',nonzero=True)
j = sy.Symbol('j',real=True)
m = sy.Symbol('m',real=True)

# perturbation H1
def fill_Q(i,j):
    if i==j:
        return Qd[j]
    if i-j==2:
        return Ql[i]
    if j-i==2:
        return Qu[i]
    return 0
Qjj = sy.Matrix(5,5,fill_Q)
h1 = sy.Matrix.diag([aperp]*5) + sy.Matrix.diag([da]*5)*Qjj
print('H1/E**2-------------------------------')
print(h1)
H1 = E**2*h1

# equilibrium density
rho0 = sy.Matrix.diag(sy.symbols('Rho0:5'))

# perturbed density
# rhod = sy.symbols('rhod0:5')
rhou = sy.symbols('rhou0:5')
rhol = sy.symbols('rhol0:5')
def fill_rho1(i,j):
    if i-j==2:
        return rhol[i]
    if j-i==2:
        return rhou[i]
    return 0
rho1_0 = sy.Matrix(5,5,fill_rho1)

# equilibrium H0
h = sy.symbols('H0:5')
H0 = sy.Matrix.diag(h)

# time evolution of rho1
def drho1(rho1):
    return dt*-1j*(H1*rho0 - rho0*H1 + H0*rho1 - rho1*H0)/hbar
print('drho1-------------------------------')
print(sy.simplify(drho1(rho1_0).subs(E**2,hbar*wE/da)))

# auxiliary density
# T is rhoj - rhoj'
# W is H0j - H0j'
# when inspecting the signs from drho1 n.b. j is row, j' is column
# The eq for drho/dt below comes from inspecting drho1.
rho = sy.Function('rho')(t)
hat = sy.Function('hat')(t)
eq1 = sy.Derivative(rho,t) - (-1j*W*rho + 1j*wE*Q*T)
eq2 = rho - hat*Q*T*sy.exp(-1j*W*t)
eq3 = sy.diff(eq2,t)
dhat_dt_1 = sy.solve(eq3,sy.Derivative(hat,t))[0]
dhat_dt_2 = dhat_dt_1.subs(hat,sy.solve(eq2,hat)[0])
dhat_dt_3 = dhat_dt_2.subs(sy.Derivative(rho,t),sy.solve(eq1,sy.Derivative(rho,t))[0])
# density
drho_dt_1 = sy.solve(eq3,sy.Derivative(rho,t))[0]

print('Main Equations--------------------------')
print('dhat/dt =',sy.simplify(dhat_dt_3))
print('drho/dt =',drho_dt_1)
print('the term in drho/dt with dhat/dt vanishes from the trace by symmetry')

# expectation value of orientation
Q_exp = (Qjj*(rho0+rho1_0)).trace()
print('<Q>-------------------------------')
print(Q_exp)

# perform the sum over Q^2
print('off-diagonal sum')
print('----------------')
summation_num = sy.factor(sy.Sum((j**2-m**2)*((j-1)**2-m**2),(m,-j,j)).doit())
summation = summation_num/(((2*j-1)**2)*(2*j+1)*(2*j-3))
sy.pprint(sy.simplify(summation))

print('diagonal sum')
print('------------')
sum1 = sy.factor(sy.Sum((j**2-m**2)**2,(m,-j,j)).doit())
sum2 = sy.factor(sy.Sum(2*(j**2-m**2)*((j+1)**2-m**2),(m,-j,j)).doit())
sum3 = sy.factor(sy.Sum(((j+1)**2-m**2)**2,(m,-j,j)).doit())
summation = sum1/((2*j-1)*(2*j+1))**2 + sum2/((2*j-1)*(2*j+1)*(2*j+1)*(2*j+3)) + sum3/((2*j+1)*(2*j+3))**2
sy.pprint(sy.simplify(summation))
