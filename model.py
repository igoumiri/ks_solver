"""
KS equation.

"""

from numpy import *
from scipy.fftpack import fft, ifft


class ks():
	"""This class models the following equations:

	u_t + u_xx + nu.u_xxxx + u.u_x = 0
	B.C.s : u(t,0) = u(t,L)
	        u_x(t,0) = u_x(t,L)
	I.C. : u(0,x) = u0(xi)

	on the domain x in (0,L].
	The spatial derivatives are computed in Fourier space using FFT.
	"""
	def __init__(self, p):
		N = p.N
		if N % 2 != 0:
			raise ValueError("N must be even.")
		
		k = zeros(N)
		k[0:N/2] = arange(N/2)
		k[N/2+1:] = arange(-N/2+1,0)
		
		# Spectral linear operator
		self.linop = k**2 - p.nu * k**4
		
		def pad(uh):
			"Pad Fourier coefficients with zeros at high wavenumbers for dealiasing by the 2/3 rule"
			Nf = N * 3 / 2
			uh_pad = zeros(Nf, 'complex')
			uh_pad[:N/2] = uh[:N/2]
			uh_pad[N+1:] = uh[N/2+1:]
			# Split the largest wavenumber among N/2 and -N/2
			uh_pad[N/2] = 0.5 * uh[N/2]
			uh_pad[N] = 0.5 * uh[N/2]
			return 1.5 * uh_pad
		
		def crop(uh):
			"Crop highest 1/3 of wavenumbers for dealiasing"
			uh_crop = zeros(N, 'complex')
			uh_crop[:N/2] = uh[:N/2]
			uh_crop[N/2+1:] = uh[N+1:]
			# Double the N/2 wavenumber since the range is assymetric
			uh_crop[N/2]= 2.0 * uh[N/2]
			return uh_crop * 2.0 / 3.0
		
		def nlinop(t, uh):
			"Spectral non-linear operator: u.u_x"
			uh_x = 1.j * k * uh # First derivative
			if p.dealiasing:
				uh_pad = pad(uh)
				uh_x_pad = pad(uh_x)
				u = real(ifft(uh_pad))
				u_x = real(ifft(uh_x_pad))
				return crop(fft(-u*u_x))
			else:
				u = real(ifft(uh))
				u_x = real(ifft(uh_x))
				return fft(-u*u_x)
		self.nlinop = nlinop


class ks_linear():
	"""This class models the following equations:

	u_t + u_xx + nu.u_xxxx + ul.u_x + u.ul_x = 0
	B.C.s : u(t,0) = u(t,L)
	        u_x(t,0) = u_x(t,L)
	I.C. : u(0,x) = u0(xi)

	on the domain x in (0,L] where ul is the stable solution.
	The spatial derivatives are computed in Fourier space using FFT.
	"""
	def __init__(self, p):
		N = p.N
		if N % 2 != 0:
			raise ValueError("N must be even.")
		
		k = zeros(N)
		k[0:N/2] = arange(N/2)
		k[N/2+1:] = arange(-N/2+1,0)
		
		# Spectral linear operator
		self.linop = k**2 - p.nu * k**4
		
		def pad(uh):
			"Pad Fourier coefficients with zeros at high wavenumbers for dealiasing by the 2/3 rule"
			Nf = N * 3 / 2
			uh_pad = zeros(Nf, 'complex')
			uh_pad[:N/2] = uh[:N/2]
			uh_pad[N+1:] = uh[N/2+1:]
			# Split the largest wavenumber among N/2 and -N/2
			uh_pad[N/2] = 0.5 * uh[N/2]
			uh_pad[N] = 0.5 * uh[N/2]
			return 1.5 * uh_pad
		
		def crop(uh):
			"Crop highest 1/3 of wavenumbers for dealiasing"
			uh_crop = zeros(N, 'complex')
			uh_crop[:N/2] = uh[:N/2]
			uh_crop[N/2+1:] = uh[N+1:]
			# Double the N/2 wavenumber since the range is assymetric
			uh_crop[N/2]= 2.0 * uh[N/2]
			return uh_crop * 2.0 / 3.0
		
		ul = p.ul
		ulh = fft(ul)
		ulh_x = 1.j * k * ulh
		if p.dealiasing:
			ulh_pad = pad(ulh)
			ulh_x_pad = pad(ulh_x)
			ul = real(ifft(ulh_pad))
			ul_x = real(ifft(ulh_x_pad))
		else:
			ul_x = real(ifft(1.j * k * ulh))
		
		def nlinop(t, uh):
			"Spectral non-linear operator linearized: u.u_x"
			uh_x = 1.j * k * uh # First derivative
			if p.dealiasing:
				uh_pad = pad(uh)
				uh_x_pad = pad(uh_x)
				u = real(ifft(uh_pad))
				u_x = real(ifft(uh_x_pad))
				return crop(fft(- ul * u_x - u * ul_x))
			else:
				u = real(ifft(uh))
				u_x = real(ifft(uh_x))
				return fft(- ul * u_x - u * ul_x)
		self.nlinop = nlinop


