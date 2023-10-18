#!/usr/bin/env python
# coding: utf-8

# In[65]:


#added libraries
import numpy as np
import cvxpy 
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib
import warnings
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")


# In[66]:


#solution to optimal evacuation problem
from opt_evac_data import *
def opt_evac(A, Q, F, q1, r, s, rtild, stild, T):
    n,m = A.shape
    q = cvx.Variable(n,'T')
    f = cvx.Variable(m,'T-1')
    node_risk = q.T*r + cvx.square(q).T*s
    edge_risk = cvx.vstack(cvx.abs(f).T*rtild + cvx.square(f).T*stild,)
    risk = node_risk + edge_risk
    constr = [q[:,0] == q1,
    q[:,1:] == A*f + q[:,:-1],
    0 <= q, q <= np.tile(Q,(T,1)).T,
    cvx.abs(f) <= np.tile(F,(T-1,1)).T]
    p = cvx.Problem(cvx.Minimize(sum(risk)), constr)
    p.solve(verbose=True, solver=cvx.ECOS)
    arr = lambda _: np.array(_.value)
    q, f, risk, node_risk = map(arr, (q, f, risk, node_risk))
    print("Total risk: ", p.value)
    print("Evacuated at t =", (node_risk <= 1e-4).nonzero()[0][0] + 1)
    return q, f, risk, node_risk
# solve
q, f, risk, node_risk = opt_evac(A, Q, F, q1, r, s, rtild, stild, T)
# plot
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 20})
fig, axs = plt.subplots(1,3,figsize=(15,5))
axs[0].plot(np.arange(1,T+1), risk)
axs[0].set_ylabel('$R_t$')
axs[1].plot(np.arange(1,T+1), q.T)
axs[1].set_ylabel('$q_t$')
axs[2].plot(np.arange(1,T), f.T)
axs[2].set_ylabel('$f_t$')
for ax in axs:
    ax.set_xlabel('$t$')
    if matplotlib.get_backend().lower() in ['agg', 'macosx']:
        fig.set_tight_layout(True)
    else:
        fig.tight_layout()
#plt.tight_layout()
fig.savefig('opt_evac.pdf')
fig.savefig('opt_evac.eps')


# In[67]:


#Simulation_Question 2
from blend_design_data import *
theta = cvx.Variable(k)
constraints = [np.log(P)*theta <= np.log(P_spec)]
constraints += [np.log(D)*theta <= np.log(D_spec)]
constraints += [np.log(A)*theta <= np.log(A_spec)]
constraints += [cvx.sum(theta)==1, theta>=0]
cvx.Problem(cvx.Minimize(0),constraints).solve()
w = np.exp(np.log(W)*theta.value)
print(w)
print(theta.value)


# In[82]:


#Simulation_Question 3
n1 = 1
n2 = 1
n3 = 1
n = n1 + n2 + n3
S = np.ones((n1+n2, n1+n2))
T = np.ones((n2+n3, n2+n3))
R = cp.Variable((n, n), PSD=True)
R1 = R[0:2, 0:2]
R2 = R[1:3, 1:3]
R_13 = R[0, 2]
objective = cp.Minimize(
cp.norm(R1 - S, "fro")**2 + cp.norm(R2 - T, "fro")**2 +
cp.norm(R_13, "fro")**2)
p = cp.Problem(objective, [])
p.solve()
print("Optimal value: %f" % p.value)
print("covariance matrix R\n%s" % str(R.value))
print("Eigenvalues\n%s" % str(np.linalg.eig(R.value)[0]))
R_simple = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
print("Matrix simple method\n%s" % str(R_simple))


# In[87]:


#Simulation_Question 4
from various_obj_regulator_data import *
u = cp.Variable((m, T))
x = cp.Variable((n, T+1))
objects = [
    (cp.Minimize(cp.sum_squares(u)), "a"), (cp.Minimize(cp.sum(cp.norm(u, 2, axis=0))), "b"),
    (cp.Minimize(cp.max(cp.norm(u, axis=0))), "c"), (cp.Minimize(cp.sum(cp.norm(u, 1, axis=0))), "d")
]
plt.figure(figsize=(15, 5))
for i, obj in enumerate(objects):
    const = [x[:, -1] == np.zeros(n)]
    const.append(x[:, 0] == x_init)
    for t in range(1, T+1):
        const.append(x[:, t] == A @ x[:, t-1] + B @ u[:, t-1])
    prob = cp.Problem(obj[0], const)
    prob.solve()
    plt.subplot(2, 4, i+1)
    plt.plot(u.value.T)
    if i == 0:
        plt.ylabel("$u_t$")
    plt.title(obj[1])
    plt.grid()
    plt.xlabel("t")
    plt.subplot(2, 4, i+5)
    plt.xlabel("t")
    plt.plot(np.linalg.norm(u.value, axis=0), c="green", label="$||u||_2$")
    if i == 2:
        plt.ylim(ymax=.12, ymin=0)
    if i == 0:
        plt.ylabel("$||u_t||$")
    plt.grid()
    plt.tight_layout()
plt.show()


# In[88]:


#Simulation_Question 5
from multi_risk_portfolio_data import *
w = cvx.Variable(n)
t = cvx.Variable()
risks = [cvx.quad_form(w, Sigma) for Sigma in (Sigma_1, Sigma_2, Sigma_3, Sigma_4, Sigma_5, Sigma_6)]
risk_constraints = [risk <= t for risk in risks]
prob = cvx.Problem(cvx.Maximize(w.T*mu - gamma * t),
risk_constraints + [cvx.sum(w) == 1])
prob.solve()
print('\nweights:')
print('\n'.join(['{:.6f}'.format(weight) for weight in w.value]))
print('\ngamma_k_val:')
print('\n'.join(['{:.6f}'.format(risk.dual_value.item())
                for risk in risk_constraints]))
print('\nrisk_val:')
print('\n'.join(['{:.6f}'.format(risk.value) for risk in risks]))
print('\nworst case risk:\n{:.6f}'.format(t.value))


# In[ ]:




