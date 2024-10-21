import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants and initial conditions ---------------------------------------------
LWC   = 3E-3   # kg m-3 (constant at all levels in updraft)
w     = 5.0    # m s-1 (constant vertical wind velocity)
dt    = 0.01   # time step (s)
H     = 2000   # top cloud height (m)
cbh   = 0      # cloud base (m)
E     = 1      # collision efficiency (unity)
rho_L = 1000   # density of liquid water (kg/m3)
k     = 2200   #  cm^(1/2) s^(-1)
g     = -9.8   # acceleration due to gravity (m s^(-2))

# Initial conditions
r_n   = 0.5E-3  # initial radius (m)
m_i   = rho_L * 4/3 * np.pi * r_n**3  # initial raindrop mass (kg) 

# Lists to store results
r_list = [r_n]
H_list = [H]
t = 0
t_list = [t]

# Function to calculate terminal velocity
def g_r(r_n, k):
    """
    Calculate terminal velocity for given radius and constant k.
    """
    return k * np.sqrt(r_n*1E2) *1E-2 # m/s

# Condensation: Assuming constant supersaturation of 1%
#> If only condensation, then r dr/dt = (S-1)x Z
#> Z = 70 um^2 s-1
zeta = 70 * 1E-12     # m^2 s^-1
S = 1.01             # 1% of saturation ratio

def conden(r, S, zeta):
    """
    Calculate the change in radius due to condensation.
    
    Args:
        r (float): Current radius (m).
        S (float): Saturation ratio.
        zeta (float): Condensation coefficient (m^2/s).
    
    Returns:
        float: Change in radius (m/s).
    """
    return (S - 1) * zeta  / r

# Simulation loop
condensation = False
evaporation = False

while H > cbh:
    v_t = g_r(r_n, k)  # Terminal velocity
    
    if 0 <= H <= 2000:
        v0 = w - v_t
        vf = v0 + g * dt
    else:
        v0 = - v_t
        vf = v0 + g * dt
    
    # Update mass using the growth equation
    m_ip1 = m_i + dt * (E * np.pi * r_n**2 * v_t * LWC)
    
    # Calculate new radius from new mass
    r_ip1 = (3 * m_ip1 / (4 * np.pi * rho_L)) ** (1/3)
    
    # Update radius due to condensation
    if condensation:
        r_ip1 += conden(r_n, S, zeta)*dt
    else:
        pass
    
    # Updating radius and mass
    r_n = r_ip1
    m_i = m_ip1
    r_list.append(r_ip1)

    # Update height using kinematic formula
    H += v0 * dt + 0.5 * g * dt**2
    H_list.append(H)
    
    # Update time
    t += dt
    t_list.append(t)

# Results
print(f"Final radius after falling from {H_list[0]} meters: {r_n*1000:.5f} mm")
print(f"Total time taken: {t/3600:.2f} hours")

if condensation:
    df2 = pd.DataFrame({'r': r_list, 'H': H_list, 't': t_list}).set_index('t')
    df2.r *= 1000  # Convert radius to mm for the plot
else:
    df1 = pd.DataFrame({'r': r_list, 'H': H_list, 't': t_list}).set_index('t')
    df1.r *= 1000  # Convert radius to mm for the plot

# Plotting results
fig, ax1 = plt.subplots()

# Primary axis for Height
df1['H'].plot(ax=ax1, color='tab:blue')
ax1.set_ylabel('Cloud Height (m)', color='tab:blue')
ax1.set_xlabel('Time (seconds)')

# Secondary y-axis for Radius
ax2 = ax1.twinx()
df1['r'].plot(ax=ax2, color='tab:red', label='r - Neglected')
df2['r'].plot(ax=ax2, style = '-.r', label='r - Condensation')
ax2.set_ylabel('Drop Radius (mm)', color='tab:red')
ax2.legend()

# Adding annotations
ax1.text(100, 0, f"Final drop radius: {df1.iloc[-1,0]:.2f} mm\nTime: {df1.index[-1]/60:.2f} min\nΔt = {dt} seconds.")
fig.tight_layout()
fig.savefig('fig/plot_grd.png', facecolor = 'white')
plt.show()