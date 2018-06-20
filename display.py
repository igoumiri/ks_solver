"""
Display solution of the KS equation as specified in the parameters.

"""

from matplotlib.pyplot import *
from scipy.fftpack import fft, ifft
from scipy import real
from scipy import linspace


def showResults(uh, p):
 	"Plot results as specified in the parameters"
 	
 	x = linspace(0, p.L, p.N)
	u = real(ifft(uh)) # Potential trap: maybe uh[p.N/2] = 0.5 *uh[p.N/2]
	
	# Colormap
	if p.show_colormap:
		figure()
		imshow(u, interpolation='bilinear', cmap=cm.gray, origin='lower', \
				aspect='auto', extent=[0, p.L, 0, p.T])
		title('Solution of the KS equation, %d modes, nu=4/%f' % (p.N, p.alpha))
		xlabel('x')
		ylabel('t')
	
	# Initial
	if p.show_initial:
		figure()
		plot(x,u[0])
		title("Initial condition u_0(x)")
		xlabel('x')
		ylabel('u_0(x)')
	
	# Final
	if p.show_final:
		figure()
		plot(x,u[-1])
		title("u(x) at T=%.2f, %d modes, nu=4/%f" % (p.T, p.N, p.alpha))
		xlabel('x')
		ylabel('u(x)')
	
	# Diff
	if p.show_diff:
		figure()
		plot(x,u[-1]-p.ul)
		title("Difference between u(x) and u_l(x) at T=%.2f, %d modes, nu=4/%f" % (p.T, p.N, p.alpha))
		xlabel('x')
		ylabel('u(x)-ul(x)')
	
	show()

