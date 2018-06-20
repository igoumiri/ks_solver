"""
Save results in either CSV or numpy's format as specified in the parameters.

"""

from numpy import save, savetxt, load
from scipy.fftpack import ifft
from scipy import real


def saveResults(uh, p):
	"Save results in either CSV or numpy's format as specified in the parameters"
	
	 # Potential trap: maybe uh[p.N/2] = 0.5 *uh[p.N/2] before ifft
	if p.save_format_csv:
		if p.save_u:
			savetxt('u.csv', real(ifft(uh)), delimiter=',')
		if p.save_uh:
			savetxt('uh.csv', uh, delimiter=',')
		if p.save_u0:
			savetxt('u0.csv', real(ifft(uh[0])), delimiter=',')
		if p.save_ul:
			savetxt('ul.csv', real(ifft(uh[-1])), delimiter=',')
		if p.save_up:
			savetxt('up.csv', p.up, delimiter=',')
	else:
		if p.save_u:
			save('u.npy', real(ifft(uh)))
		if p.save_uh:
			save('uh.npy', uh)
		if p.save_u0:
			save('u0.npy', real(ifft(uh[0])))
		if p.save_ul:
			save('ul.npy', real(ifft(uh[-1])))
		if p.save_up:
			save('up.npy', p.up)

