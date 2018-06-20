"""
Spectral Collocation integrator.

"""

class spectralCollocation():
	def __init__(self, model, p):
		
		A = (1 + 0.5 * p.dt0 * model.linop) / (1 - 0.5 * p.dt0 * model.linop)
		B = p.dt0 / (1 - 0.5 * p.dt0 * model.linop)
		def stepInit(t, uh):
			return A * uh + B * model.nlinop(t, uh)
		self.stepInit = stepInit
		
		
		
		A = (1 + 0.5 * p.dt * model.linop) / (1 - 0.5 * p.dt * model.linop)
		B = p.dt / (1 - 0.5 * p.dt * model.linop)
		def step(t, uh, uh_pre):
			return A * uh + B * (3.0 * model.nlinop(t, uh) - model.nlinop(t, uh_pre)) / 2.0
		self.step = step

