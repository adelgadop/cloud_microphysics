import numpy as np
import matplotlib.pyplot as plt

# Namelist ---------------------------------------------------------------------
U       = 1     # m/s
t_end   = 60    # seconds
dt      = 0.001 # time-step in seconds
x, dx   = np.linspace(0, 20, 201, retstep = True)   # mesh spacing of 0.1 metres

def advection(U, t_end, dt, x):
    # Initial conditions
    conc    = 0.001 # kg of liquid per kg of air
    t_start = 0     # 0 seconds
        
    cn      = np.where(x <= 1, conc, 0) # Concentrations over the first metre
    cnp1    = cn.copy()*0
    CFL     = U*dt/dx
    result  = [cn.copy()]
    
    while t_start < t_end:
        cnp1[1:] = cn[1:] - CFL * (cn[1:] - cn[:-1]) 

        # saving result of looping
        result.append(cnp1.copy())
        cn      = cnp1.copy()

        t_start += dt
        
    return result

res1 = advection(1, t_end, dt, x)
res2 = advection(10, t_end, dt, x)
res3 = advection(50, t_end, dt, x)
res4 = advection(100, t_end, dt, x)

fig, axes = plt.subplots(4,1, figsize= (6,7), sharex=True)
for ax, res, U in zip(axes.flatten(),[res1, res2, res3, res4], [1, 10, 50, 100]):
    for n in [0, 100, 500, 1000]:
        ax.plot(x, res[n], label=f't={n*dt:.2f} s')
    
    if (U == 10) | (U == 50):
        ax.text(8, 0.00085, "$\\vec{U}$" + f" = {U} m/s")
        ax.text(8, 0.0007, f"CFL = {U*dt/dx:.2f}")
        ax.text(8, 0.00055, f"$\Delta t$ = {dt:.3f}")
        
    else:
        ax.text(3, 0.00085, "$\\vec{U}$" + f" = {U} m/s")
        ax.text(3, 0.0007, f"CFL = {U*dt/dx:.2f}")
        ax.text(3, 0.00055, f"$\Delta t$ = {dt:.3f}")
        
axes[0].legend(ncol=1)
axes[0].set_title('Advection of Cloud with 1st order of approximation', loc='left')
axes[3].set_xlabel('Distance in meters')
fig.text(-0.02, 0.5, 'Concentration (kg of liquid/kg of air)', va='center', rotation='vertical')
fig.savefig('fig/plot_adv_exercise3.png', facecolor = 'white')
plt.show()


def advection_2nd_order(U, t_end, dt, x):
    """_summary_

    Args:
        U (float): wind velocity
        t_end (float): end time
        dt (float): time step
        x (array): a spatial vector of 1D dimension 

    Returns:
        _type_: results with numerical integrations
    """
    # Initial conditions
    conc    = 0.001 # kg of liquid per kg of air
    t_start = 0     # 0 seconds
        
    cnm1    = np.where(x <= 1, conc, 0)   # C_{n-1}
    cn      = cnm1.copy()*0               # C_{n}
    cnp1    = cnm1.copy()*0               # C_{n+1}

    CFL     = U*dt/dx
    result  = [cnm1.copy()]
    
    # First order Euler-Forward for 1st iteration
    cn[1:] = cnm1[1:] - CFL * (cnm1[1:] - cnm1[:-1]) 

    # saving result
    result.append(cn.copy())
    t_start += dt
    
    while t_start < t_end:
        cnp1[1:] = cnm1[1:] - CFL * (cn[1:] - cn[:-1])
        
        # saving result of looping
        result.append(cnp1.copy())
        cnm1, cn = cn, cnp1
        t_start += dt            
        
    return result

res1 = advection_2nd_order(1, t_end, dt, x)
res2 = advection_2nd_order(10.0, t_end, dt, x)
res3 = advection_2nd_order(15.0, t_end, dt, x)
res4 = advection_2nd_order(100, t_end, dt, x)

fig, axes = plt.subplots(4,1, figsize= (6,7), sharex=True)
for ax, res, U in zip(axes.flatten(),[res1, res2, res3, res4], [1, 10, 50, 100]):
    for n in [0, 100, 500, 1000]:
        ax.plot(x, res[n], label=f't={n*dt:.2f} s')
        
    if U == 100:
        ax.text(3, 0.0012, "$\\vec{U}$" + f" = {U} m/s")
        ax.text(3, 0.0007, f"CFL = {U*dt/dx:.2f}")
        ax.text(3, 0.0002, f"$\Delta t$ = {dt:.3f}")
        
    else: 
        ax.text(3, 0.00085, "$\\vec{U}$" + f" = {U} m/s")
        ax.text(3, 0.0007, f"CFL = {U*dt/dx:.2f}")
        ax.text(3, 0.00055, f"$\Delta t$ = {dt:.3f}")

axes[0].legend(ncol=1)
axes[0].set_title('Advection of Cloud with Leap-Frog (2nd order approx.)', loc='left')
axes[3].set_xlabel('Distance in meters')
fig.text(-0.02, 0.5, 'Concentration (kg of liquid/kg of air)', va='center', rotation='vertical')
fig.savefig('fig/plot_adv_2nd_order_exercise3.png', facecolor = 'white')
plt.show()