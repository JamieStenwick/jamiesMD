/* Jamie Stenwick; as simple of a molecular dynamics simulation as one can get.
Assumptions / Current Conditions:
Lennard-Jones Potential, V(r) = 4*eps[(sigma/r)^12 - (sigma/r)^6]
All radii = 1 (reference length)
All mass = 1 (reference mass)
The 3 indpendent units we will choose are length, energy, and mass
L_0, E_0, M_0 are the reference length, energy, and mass respectively, in our simulation
we take L_0 = a, a = particle radius, E_0 = kT_ref, M_0 = m_0, m_0 = particle mass
Non-dimensional Drag = 1 (not a reference value, gamma * L_0 / sqrt(M_0 * E_0))
Non-dimensional time unit [dt] = a*sqrt(m_0 / kT_ref)
r distance coordinate is center-center in radii
Translational DOF only
Cell list
*/
