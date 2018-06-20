#!/usr/bin/env python
"""
Solver for the non-linear and linearized Kuramoto-Sivashinsky (KS) equation.

Imene Goumiri
MIT License
Princeton University
"""

import sys
from numpy import * # So we don't have to prefix every other command...
from numpy.random import random_sample
from scipy.fftpack import fft

from model import ks, ks_linear
from integrator import spectralCollocation
from display import showResults
from save import saveResults


class parameters():
	"Default parameters"

	# Run linearized code
	linear = False

	# Length of the domain
	L = 2 * pi

	# Number of points in the subdivision
	N = 128

	# Bifurcation parameters
	alpha = 7.916
	nu = 4.0 / alpha

	# Dealiasing for FFT operations
	dealiasing = True

	# Final time
	T = 100

	# Time step for the first step
	dt0 = 0.0001

	# Time step for all other steps
	dt = 0.01

	# Initial condition

	ul = load('ul.npy')
	#up = load('up.npy')
	#up = 0.001 * random_sample(N)
	#u0 = 0.2*random_sample(N)
	up = 0.1*sin(linspace(0, L, N))
	up[N/2] = 0
	#u0 = up
	u0 = ul
	#up = 0.1*sin(linspace(0, L, N))

	# Stable solution around which we linearize
	if linear:
		try:
			ul = load('ul.npy')
		except:
			ul = loadtxt('ul.csv', delimiter=',')

	# Display the results
	show_colormap = True
	show_initial = True
	show_final = True
	show_diff = True
	show_any = any([show_colormap, show_initial, show_final, show_diff])

	# Save the results
	save_u = False
	save_uh = False
	save_u0 = False
	save_up = False
	save_ul = False # Final step, to use in the linearization
	save_format_csv = False # True for CSV, False for numpy's binary format
	save_any = any([save_u, save_uh, save_ul, save_u0, save_up])




if __name__ == '__main__':
	p = parameters()

	if p.linear:
		model = ks_linear(p)
	else:
		model = ks(p)
	integrator = spectralCollocation(model, p)

	uh = zeros((len(arange(0, p.T, p.dt))+1, p.N), 'complex')
	uh[0] = fft(p.u0)

	# First time step
	uh[1] = uh[0]
	for t in arange(0, p.dt, p.dt0):
		uh[1] = integrator.stepInit(t, uh[1])

	# Other time steps
	for k, t in enumerate(arange(p.dt, p.T, p.dt), 1):
		uh[k+1] = integrator.step(t, uh[k], uh[k-1])
		if isnan(uh[k+1]).any():
			raise RuntimeError("Integration stopped, NaN encountered at iteration %d" % (k+1))

	if p.show_any:
		showResults(uh, p)

	if p.save_any:
		saveResults(uh, p)

