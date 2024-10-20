# ==============================================================================
# Cloud microphysics functions
# Created by Alejandro D. Peralta
# Last version Oct. 20 2024
# ==============================================================================

# Libraries --------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Namelist ---------------------------------------------------------------------
dt        = 5                # Time step in seconds
T0        = 288.15           # Initial temperature at sea level (K)
w         = 5                # Ascent rate in m/s
p0        = 101325           # Sea level pressure (Pa)
z0        = 0                # Initial altitude in meters
Gamma_dry = 9.8 / 1000       # Dry adiabatic lapse rate in K/m
Gamma_sat = 6.0 / 1000       # Saturated adiabatic lapse rate in K/m
H = 7300                     # Scale height (m)
E = 0.622                    # Ratio water vapour and dry air

# Ascent rates (m/s)
ascent_rates = [1, 10, 30]
is_saturated = True

def saturation_vapor_pressure(T):
    A = 2.53*1E11            # For Eq. 3 (Rogers and Yau, 1996) (Pa)
    B = 5.42*1E3             # For Eq. 3 (Rogers and Yau, 1996)
    return A * np.exp(-B/T)  # Eq. 3

def simulate_ascent_temp(ascent_rate):
    z = np.arange(z0, 5000 + dt*ascent_rate, dt*ascent_rate)
    # Initialize altitude, temperature, and pressure arrays
    Tsat   = np.zeros_like(z, dtype=float)
    Tdry   = np.zeros_like(z, dtype=float)
    p   = np.zeros_like(z, dtype=float)

    # Set initial conditions
    Tsat[0]   = T0
    Tdry[0]   = T0
    p[0]      = p0
    
    for i in range(1, len(z)):
        # Update pressure using the exponential formula
        p[i] = p0 * np.exp(-z[i]/H)
        Tsat[i] = Tsat[i-1] - Gamma_sat * (z[i] - z[i-1])
        Tdry[i] = Tdry[i-1] - Gamma_dry * (z[i] - z[i-1])
            
    return z, Tsat, Tdry, p

# US  Standard Atmosphere vs Altitude ------------------------------------------
data = {
    "Altitude (m)": [-1000, 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 80000],
    "Temperature (°C)": [21.50, 15.00, 8.50, 2.00, -4.49, -10.98, -17.47, -23.96, -30.45, -36.94, -43.42, -49.90, -56.50, -56.50, -51.60, -46.64, -22.80, -2.50, -26.13, -53.57, -74.51],
    "Gravity (m/s²)": [9.810, 9.807, 9.804, 9.801, 9.797, 9.794, 9.791, 9.788, 9.785, 9.782, 9.779, 9.776, 9.761, 9.745, 9.730, 9.715, 9.684, 9.654, 9.624, 9.594, 9.564],
    "Pressure (10⁴ N/m²)": [11.39, 10.13, 8.988, 7.950, 7.012, 6.166, 5.405, 4.722, 4.111, 3.565, 3.080, 2.650, 1.211, 0.5529, 0.2549, 0.1197, 0.0287, 0.007978, 0.002196, 0.000502, 0.000111],
    "Density (kg/m³)": [1.347, 1.225, 1.112, 1.007, 0.9093, 0.8194, 0.7364, 0.6601, 0.5900, 0.5258, 0.4671, 0.4135, 0.1948, 0.08891, 0.04008, 0.01841, 0.003996, 0.001027, 0.0003097, 0.00008283, 0.00001846],
    "Viscosity (10⁻⁵ N s/m²)": [1.821, 1.789, 1.758, 1.726, 1.694, 1.661, 1.628, 1.595, 1.561, 1.527, 1.493, 1.458, 1.422, 1.422, 1.448, 1.475, 1.601, 1.704, 1.584, 1.438, 1.321]
}

# Convert to a pandas DataFrame
df = (pd.DataFrame(data)).iloc[1:,:]
df['Pressure (10⁴ N/m²)'] *= 100
df.rename(columns={'Pressure (10⁴ N/m²)': 'Pressure (hPa)'}, inplace = True)
df.drop(['Gravity (m/s²)', 'Viscosity (10⁻⁵ N s/m²)'], axis=1, inplace = True)

# Comparison between calculated temperatures and US Standard Atmosphere (1976)
z, Tsat, Tdry, p = simulate_ascent_temp(w)

df2 = pd.DataFrame({'Altitude (m)':z, 'Temp. Dry (°C)':Tdry - 273.15, 
              'Temp. Sat. (°C)':Tsat - 273.15, 'Pressure (hPa) (calc.)':p/100})

df_all = df.merge(df2, on='Altitude (m)')

# Plot altitude, temperature, and pressure over time
fig, ax = plt.subplots(1, figsize=(7,6))

# Plot altitude vs temperature vs pressure
ax.set_xlabel('Temperature (°C)', color ='b')
ax.set_ylabel('Altitude (m)')
ax.plot(df_all['Temp. Sat. (°C)'], df_all['Altitude (m)'], color ='b', marker = 's', label = 'T$_{sat}$ (calculated)', markersize=6, markeredgecolor ='k')
ax.plot(df_all['Temp. Dry (°C)'], df_all['Altitude (m)'], color ='b', marker='o', label = 'T$_{dry}$ (calculated)', markersize=6, markeredgecolor ='k')
ax.plot(df_all['Temperature (°C)'], df_all['Altitude (m)'], color ='b', marker='^', label = 'Temp. (Std. Atm.) ', markersize=6, markeredgecolor ='k')
ax.tick_params(axis='x', colors = 'b')

# Create a secondary x-axis for pressure
axt = ax.twiny()
axt.plot(df_all['Pressure (hPa)'], df_all['Altitude (m)'], 'ro-', label="Pressure (hPa) (Std. Atm.)")
axt.plot(df_all['Pressure (hPa) (calc.)'], df_all['Altitude (m)'], 'r^-', label="Pressure (hPa) (calculated)")
axt.set_xlabel('Pressure (hPa)', color='r')
axt.tick_params('x', colors='r')


# Plot pressure on a third axis
ax.legend(loc='upper right')

plt.legend(loc='lower left')
# Plot pressure on a third axis
fig.tight_layout()
fig.savefig('fig/plot_temp_profile.png', facecolor = 'white')
plt.show()

# ==============================================================================
# Including water vapor, droplets and rain mixing ratio
# ==============================================================================
# Namelist ---------------------------------------------------------------------
dt        = 5                      # Time step in seconds
T0        = 288.15                 # Initial temperature at sea level (K)
w         = 5                      # Ascent rate in m/s
H         = 7300                   # Scale height (m)
z0        = 100                    # Initial altitude in meters
p0        = 101325*np.exp(-100/H)  # Pressure level at initial altitude (Pa)
Gamma_dry = 9.8 / 1000             # Dry adiabatic lapse rate in K/m
T0        = T0 - Gamma_dry * z0    # Temperature at first level
Gamma_sat = 6.0 / 1000             # Saturated adiabatic lapse rate in K/m
E         = 0.622                  # Ratio of molecular weights of water vapor to dry air
rh0       = 0.7                    # relative humidity
C_green   = 100                    # Green-ocean aerosol concentration (cm^-3)
C_pol     = 3000                   # Polluted aerosol concentration (cm^-3) in São Paulo
k         = 0.5                    # Twomey constant for supersaturation
Qw_crit   = 0.001                  # Critical cloud-liquid mixing ratio (Klemp 1978)
tau       = 1000                   # Fallout time in seconds
k1        = 0.001                  # Autoconversion constant in s-1
k2        = 2.2                    # Accretion constant  in s-1  

# Ascent rates (m/s)
ascent_rates = [1, 10, 30]

def simulate_ascent_all(ascent_rate, aer_conc):
    z = np.arange(z0, 5000 + dt*ascent_rate, dt*ascent_rate)

    T   = np.zeros_like(z, dtype=float)
    p   = np.zeros_like(z, dtype=float)
    qv  = np.zeros_like(z, dtype=float)
    qvs = np.zeros_like(z, dtype=float)
    qw  = np.zeros_like(z, dtype=float)
    qt  = np.zeros_like(z, dtype=float)
    e   = np.zeros_like(z, dtype=float)
    es  = np.zeros_like(z, dtype=float)
    RH  = np.zeros_like(z, dtype=float)
    nw  = np.zeros_like(z, dtype=float)  # Number concentration of droplets
    qr  = np.zeros_like(z, dtype=float)  # Concentration of rain mixing ratio
    ssat= np.zeros_like(z, dtype=float)  # Supersaturation

    # Set initial conditions
    T[0]   = T0
    p[0]   = p0
    es[0]  = saturation_vapor_pressure(T0)
    e[0]   = rh0*es[0]
    qv[0]  = E*e[0]/(p0 - e[0])
    qt[0]  = qv[0]
    qvs[0] = E * es[0]/(p[0]-es[0])
    RH[0]  = rh0
    ssat[0] = e[0]/es[0] - 1 if RH[0] > 1 else 0
    nw[0] = aer_conc * ssat[0] ** k if RH[0] >= 1 else 0  # Twomey's formula
    
    for i in range(1, len(z)):
        # Update pressure using the exponential formula
        p[i] = p0 * np.exp(-z[i]/H)                               # rev. ok
        qt[i] = qt[0]                                             # rev. ok
        qvs[i] = E * es[i-1]/(p[i-1] - es[i-1])
        qv[i]  = E * e[i-1] /(p[i-1] - e[i-1] )

        if qv[i] < qvs[i]:  # unsaturated
            T[i]  = T[i-1] - Gamma_dry * (z[i] - z[i-1])          # rev. ok
            qv[i] = qv[0]                                         # rev. ok
            qw[i] = 0                                             # rev. ok
            e[i]  = qv[i]/(qv[i] + E) * p[i]                      # rev. ok
            es[i] = saturation_vapor_pressure(T[i])               # rev. ok
            #qvs[i] = E * es[i]/(p[i] - es[i])
            RH[i] = e[i] / es[i]
                    
        elif qv[i] >= qvs[i]: # saturated
            T[i] = T[i-1] - Gamma_sat * (z[i] - z[i-1])           # rev. ok
            qv[i] = qvs[i]                                        # rev. ok
            qw[i] = qt[i] - qvs[i]                                # rev. ok
            e[i]  = qv[i]/(qv[i] + E) * p[i]                      # rev. ok
            es[i] = saturation_vapor_pressure(T[i])               # rev. ok
            #qvs[i] = E * es[i]/(p[i] - es[i])
            ssat[i] = 100*(e[i]/es[i] - 1)                        # rev. ok
            RH[i] = e[i] / es[i]                                  # rev. ok
            
            # Include droplet number mixing ratio n_w             # Under review
            if nw[i] > 0:
                nw[i] = nw[i-1]
            elif nw[i] == 0:
                nw[i] = aer_conc * ssat[i] ** k
            
            # Rain formation                                      # Under review
            #> Accretion (C_r)
            #> Autoconversion (A_r)
            A_r   = k1 * (qw[i] - Qw_crit) if qw[i] > Qw_crit else 0  # Autoconversion
            C_r   = k2 * qr[i-1] * qw[i]  # Accretion
            F     = qr[i-1] / tau         # Fallout
            qr[i] = qr[i-1] + dt * (A_r + C_r - F)  # Update rain mixing ratio
            
    return z, T, p, qv, qvs, qw, qt, e, es, RH, nw, qr, ssat

# Plot results
fig, axes = plt.subplots(3,3, figsize=(10,8), sharey=True, gridspec_kw={'wspace':0.1, 'hspace':0.3})
ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = axes.flatten()
for w, ax in zip(ascent_rates, [ax1, ax2, ax3]):
    z, T, p, qv, qvs, qw, qt, e, es, RH, nwp, qrp, ssat = simulate_ascent_all(w, C_green)

    ax.plot(qv, z, color ='tab:blue', label = 'Vapor mixing\nratio ($q_v$)')
    ax.plot(qw, z, color ='blue', label = 'Cloud-liquid\nmixing ratio ($q_w$)')
    ax.plot(qt, z, color ='k', label = 'Cloud-liquid\ncontent ($q_T$)')
    ax.set_title("$\\vec{U}$" + f" = {w} m/s", loc='left')

for w, ax in zip(ascent_rates, [ax4, ax5, ax6]):
    z, T, p, qv, qvs, qw, qt, e, es, RH, nwp, qrp, ssat = simulate_ascent_all(w, C_green)  
    ax.plot(qvs, z, color ='m', label = 'Saturated mixing ratio ($q_{vs}$)')
 
for w, ax in zip(ascent_rates, [ax7, ax8, ax9]):
    z, T, p, qv, qvs, qw, qt, e, es, RH, nwp, qrp, ssat = simulate_ascent_all(w, C_green)
    ax.plot(RH, z, color ='tab:blue', label = 'Relative humidity')

ax2.set_xlabel('Mixing ratio')
ax4.set_ylabel('Altitude (meters)', color='k')

ax1.legend(loc='lower center', fontsize=6, ncol=2)
ax4.legend(loc='upper right', fontsize=6)
ax5.set_xlabel('Mixing ratio')
ax7.legend(loc='upper left', fontsize=6)
ax8.set_xlabel('Relative humidity')
plt.show()
fig.savefig('fig/plot_mixing_RH_exercise2.png', bbox_inches='tight', 
            facecolor = 'white')

# Plot droplets number and rain mixing ratio -----------------------------------
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=True, sharex=True)
ascent_rates = [1, 10, 30]
aerosol_conditions = [C_green, C_pol]

for ax, w in zip(axes[0], ascent_rates):
    z, T, p, qv, qvs, qw, qt, e, es, RH, nwg, qrg, ssat = simulate_ascent_all(w, C_green)
    ax.plot(nwg, z, label = 'Droplets number in cm$^{-3}$', color='m')
    axt = ax.twiny()
    axt.plot(qrg, z, label = 'Rain mixing ratio (kg/kg)', color ='b')
    ax.set_title(f"Ascent Rate = {w} m/s")
    ax.set_xscale('log')
    axt.set_xscale('log')
    ax.tick_params('x', colors='m')
    axt.tick_params('x', colors='b')
axes[0][1].legend(loc='lower left')
axes[0][0].set_ylabel('Altitude (m)')
plt.legend(loc='lower left')
    
for ax, w in zip(axes[1], ascent_rates):
    z, T, p, qv, qvs, qw, qt, e, es, RH, nwp, qrp, ssat = simulate_ascent_all(w, C_pol)
    ax.plot(nwp, z, color='m')
    axt = ax.twiny()
    axt.plot(qrp, z, color ='b')
    ax.set_xscale('log')
    axt.set_xscale('log')
    ax.tick_params('x', colors='m')
    axt.tick_params('x', colors='b')

axes[1][0].set_ylabel('Altitude (m)')
axes[1][1].set_xlabel('concentration')

plt.show()
fig.savefig('fig/plot_mixing_rain_formation.png', bbox_inches='tight', 
            facecolor = 'white')
