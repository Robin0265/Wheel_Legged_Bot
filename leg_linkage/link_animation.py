import numpy as np
import matplotlib.pyplot as plt

from mechanism import Mechanism, Joint, Vector

# -------------------------
# Link lengths (m)
# -------------------------
L_GROUND  = 3.0
L_INPUT   = 1.0
L_COUPLER = 3.0
L_OUTPUT  = 1.0

# -------------------------
# Joints (match the example's topology)
# O = right ground pivot
# C = left ground pivot
# A = crank end
# B = coupler-output joint
# -------------------------
O = Joint("O")  # right ground
A = Joint("A")  # crank end (moving)
B = Joint("B")  # moving
C = Joint("C")  # left ground

# Vectors (a is the input crank; c is the ground link)
a = Vector((O, A), 
           r=L_INPUT)                       # input: O->A
b = Vector((A, B), 
           r=L_COUPLER)                     # coupler: A->B
c = Vector((O, C), 
           r=L_GROUND, 
           theta=np.pi, 
           style="ground")                  # ground: O->C (to the left)
d = Vector((C, B), 
           r=L_OUTPUT)                      # output: C->B

# Input motion: full 360 deg (avoid exact toggle endpoints)
n_frames = 360
time = np.linspace(0.0, 1.0, n_frames)


# Constant angular velocity for the input crank
angular_velocity = 2 * np.pi / (time[-1] - time[0])  # rad/s
theta = angular_velocity * time
omega = np.full(n_frames, angular_velocity)
alpha = np.zeros(n_frames)

# Avoid exact 0 and 2pi to prevent singularities
eps = 1e-3
theta = np.linspace(eps, 2*np.pi - eps, n_frames)

# Guesses for the unknown angles (b and d)
pos_guess = np.deg2rad([45, 90])
vel_guess = np.deg2rad([100, 100])
acc_guess = np.deg2rad([100, 100])

# Loop equation (exact pattern used in the repo example)
# a(i) + b(x[0]) - c() - d(x[1]) = 0
# input vector should be defined here
# x[0] = angle of vector b
# x[1] = angle of vector d
def loop(x, input):
    return a(input) + b(x[0]) - c() - d(x[1])

# mechanism object
mech = Mechanism(
    vectors=(a, b, c, d),
    origin=O,
    loops=loop,
    pos=theta,
    vel=omega,
    acc=alpha,
    guess=(pos_guess, vel_guess, acc_guess)
)

# animation
mech.iterate()
ani_, fig_, ax_ = mech.get_animation()

ax_.set_title('Linkage')

# Analysis plots
fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

# Data for vector d (output link)
pos_arr = np.asarray(d.pos.thetas)
vel_arr = np.asarray(d.vel.omegas)
acc_arr = np.asarray(d.acc.alphas)

ax[0].plot(time, pos_arr, color='orange')
ax[1].plot(time, vel_arr, color='green')
ax[2].plot(time, acc_arr, color='red')

ax[0].set_ylabel(r'$\theta$')
ax[1].set_ylabel(r'$\omega$')
ax[2].set_ylabel(r'$\alpha$')

ax[2].set_xlabel(r'Time (s)')
ax[0].set_title(r'Analysis of $\vec{d}$')

for axis in ax:
    axis.minorticks_on()
    axis.grid(which='both')

plt.show()