import ctypes
import numpy.ctypeslib as ctl
import numpy as np
import os
import platform

if os.path.exists(os.environ["SIMULATION_ROUTINES_PATH"] + '/routines_lib.so') == False:
	raise ImportError('No routines_lib.so library found')
lib=ctl.load_library('routines_lib.so', os.environ["SIMULATION_ROUTINES_PATH"])

#Reward table function
reward_table_cpp = lib.reward_table
reward_table_cpp.argtypes = [ctl.ndpointer(np.float64, 
                                         flags='aligned, c_contiguous'), 
                           ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]

def expected_reward_table(win_max, fail_max, time_max, **kwargs):
	prior = (1.0, 1.0)
	cost = 0.0
	if 'prior' in kwargs:
		prior = kwargs.pop('prior')
	if 'cost' in kwargs:
		cost = kwargs.pop('cost')
	if(len(kwargs) > 0):
		raise TypeError(list(kwargs.keys())[0] + ' is an invalid keyword argument')

	val = np.zeros(win_max*fail_max*time_max, dtype=np.float64)
	reward_table_cpp(val, win_max, fail_max, time_max, cost, prior[0], prior[1])
	val = val.reshape((time_max, win_max, fail_max))
	return val

#Transition table function
transition_table_cpp = lib.transition_table
transition_table_cpp.argtypes = [ctl.ndpointer(np.intc, 
                                         flags='aligned, c_contiguous'), 
                           ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]

def optimal_transition_table(win_max, fail_max, time_max, **kwargs):
	prior = (1.0, 1.0)
	cost = 0.0
	if 'prior' in kwargs:
		prior = kwargs.pop('prior')
	if 'cost' in kwargs:
		cost = kwargs.pop('cost')
	if(len(kwargs) > 0):
		raise TypeError(list(kwargs.keys())[0] + ' is an invalid keyword argument')

	val = np.zeros(win_max*fail_max, dtype=np.intc)
	transition_table_cpp(val, win_max, fail_max, time_max, cost, prior[0], prior[1])
	val = val.reshape((win_max, fail_max))
	return val

#Truncated Bernoulli functions

truncated_reward_table_cpp = lib.truncated_reward_table
truncated_reward_table_cpp.argtypes = [ctl.ndpointer(np.float64, 
                                         flags='aligned, c_contiguous'), 
                           ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double]

def expected_truncated_reward_table(win_max, fail_max, time_max, **kwargs):
	truncation = 1.0
	cost = 0.0
	if 'truncation' in kwargs:
		truncation = kwargs.pop('truncation')
	if(len(kwargs) > 0):
		raise TypeError(list(kwargs.keys())[0] + ' is an invalid keyword argument')

	val = np.zeros(win_max*fail_max*time_max, dtype=np.float64)
	truncated_reward_table_cpp(val, win_max, fail_max, time_max, cost, truncation)
	val = val.reshape((time_max, win_max, fail_max))
	return val

truncated_transition_table_cpp = lib.truncated_transition_table
truncated_transition_table_cpp.argtypes = [ctl.ndpointer(np.intc, 
                                         flags='aligned, c_contiguous'), 
                           ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double]

def optimal_truncated_transition_table(win_max, fail_max, time_max, **kwargs):
	truncation = 0.9
	cost = 0.0
	if 'truncation' in kwargs:
		truncation = kwargs.pop('truncation')
	if(len(kwargs) > 0):
		raise TypeError(list(kwargs.keys())[0] + ' is an invalid keyword argument')

	val = np.zeros(win_max*fail_max, dtype=np.intc)
	truncated_transition_table_cpp(val, win_max, fail_max, time_max, cost, truncation)
	val = val.reshape((win_max, fail_max))
	return val

#Routine that runs arbitraty policy with arbitrary button

run_trials_cpp = lib.run_trials
run_trials_cpp.argtypes = [ctl.ndpointer(np.float64, flags='aligned, c_contiguous'), 
							ctypes.c_int, ctypes.c_double, ctypes.c_int,
							ctypes.c_char, ctl.ndpointer(np.float64, flags='aligned, c_contiguous'), 
							ctypes.c_char, ctl.ndpointer(np.float64, flags='aligned, c_contiguous')]

record_full_trials_cpp = lib.record_full_trials
record_full_trials_cpp.argtypes = [ctl.ndpointer(np.float64, flags='aligned, c_contiguous'), 
							ctypes.c_int, ctypes.c_double, ctypes.c_int,
							ctypes.c_char, ctl.ndpointer(np.float64, flags='aligned, c_contiguous'), 
							ctypes.c_char, ctl.ndpointer(np.float64, flags='aligned, c_contiguous')]

class Button(object):
	def __init__(self, type, **kwargs):
		if type == 'Bernoulli':
			self.type = 'b'
			if 'prior' in kwargs:
				self.parameters = np.array([kwargs.pop('prior')], dtype=np.float64)
			else:
				self.parameters = np.array([1.0, 1.0], dtype=np.float64)
		elif type == 'Truncated':
			self.type = 't'
			if 'truncation' in kwargs:
				self.parameters = np.array([kwargs.pop('truncation')], dtype=np.float64)
			else:
				self.parameters = np.array([0.9], dtype=np.float64)
		else:
			raise ValueError("Wrong button type")
		if(len(kwargs) > 0):
			raise TypeError(list(kwargs.keys())[0] + ' is an invalid keyword argument')


class Policy(object):
	def __init__(self, type, **kwargs):
		if type == 'Bernoulli':
			self.type = 'b'
			cost = 0.0
			prior = [1.0, 1.0]
			if 'prior' in kwargs:
				prior = kwargs.pop('prior')
			if 'cost' in kwargs:
				cost = kwargs.pop('cost')
			self.parameters = np.array([cost, prior[0], prior[1]], dtype=np.float64)
		elif type == 'Meta':
			self.type = 'm'
			if 'prior_k' in kwargs:
				self.parameters = np.array([kwargs.pop('prior_k')], dtype=np.float64)
			else:
				self.parameters = np.array([1.0], dtype=np.float64)
		elif type == 'Truncated':
			self.type = 't'
			cost = 0.0
			truncation = 0.9
			if 'cost' in kwargs:
				cost = kwargs.pop('cost')
			if 'truncation' in kwargs:
				truncation = kwargs.pop('truncation')
			self.parameters = np.array([cost, truncation], dtype=np.float64)
		elif type == 'Random':
			self.type = 'r'
			if 'switch_probability' in kwargs:
				self.parameters = np.array([kwargs.pop('switch_probability')], dtype=np.float64)
			else:
				self.parameters = np.array([0.2], dtype=np.float64)
		elif type == 'Difference':
			self.type = 'd'
			if 'difference' in kwargs:
				self.parameters = np.array([kwargs.pop('difference')], dtype=np.float64)
			else:
				self.parameters = np.array([4.0], dtype=np.float64)
		elif type == 'Ratio':
			self.type = 'f'
			if 'ratio' in kwargs:
				self.parameters = np.array([kwargs.pop('ratio')], dtype=np.float64)
			else:
				self.parameters = np.array([0.6], dtype=np.float64)
		elif type == 'Finite':
			self.type = 'n'
			cost = 0.0
			prior = [1.0, 1.0]
			if 'nbuttons' not in kwargs:
				raise TypeError('Finite Bernoulli policy is missing number of buttons parameter')
			else:
				nbuttons = kwargs.pop('nbuttons')
			if 'prior' in kwargs:
				prior = kwargs.pop('prior')
			if 'cost' in kwargs:
				cost = kwargs.pop('cost')
			self.parameters = np.array([nbuttons, cost, prior[0], prior[1]], dtype=np.float64)
		if(len(kwargs) > 0):
			raise TypeError(list(kwargs.keys())[0] + ' is an invalid keyword argument')

def run_trials(time, num_tries, button, policy, **kwargs):
	cost = 0.0
	if 'cost' in kwargs:
		cost = kwargs.pop('cost')
	if(len(kwargs) > 0):
		raise TypeError(list(kwargs.keys())[0] + ' is an invalid keyword argument')

	if(button.__class__.__name__ != 'Button'):
		raise ValueError("Wrong button entry")
	if(policy.__class__.__name__ != 'Policy'):
		raise ValueError("Wrong policy entry")

	val = np.zeros(num_tries, dtype=np.float64)
	run_trials_cpp(val, time, cost, num_tries, button.type.encode(), button.parameters, policy.type.encode(), policy.parameters)
	return val

def record_full_trials(time, num_tries, button, policy, **kwargs):
	cost = 0.0
	if 'cost' in kwargs:
		cost = kwargs.pop('cost')
	if(len(kwargs) > 0):
		raise TypeError(list(kwargs.keys())[0] + ' is an invalid keyword argument')

	if(button.__class__.__name__ != 'Button'):
		raise ValueError("Wrong button entry")
	if(policy.__class__.__name__ != 'Policy'):
		raise ValueError("Wrong policy entry")

	val = np.zeros(num_tries*time*4, dtype=np.float64)
	record_full_trials_cpp(val, time, cost, num_tries, button.type.encode(), button.parameters, policy.type.encode(), policy.parameters)
	val = val.reshape((num_tries, time, 4))
	val[::,0:2] = np.rint(val[::,0:2])
	return val

#Finite arm reward table function
finite_arm_reward_table_cpp = lib.finite_arm_reward_table
finite_arm_reward_table_cpp.argtypes = [ctl.ndpointer(np.float64, flags='aligned, c_contiguous'),
										ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
										ctypes.c_double, ctypes.c_double, ctypes.c_double]

def finite_arm_expected_reward_table(win_max, fail_max, time_max, button_max, **kwargs):
	prior = (1.0, 1.0)
	cost = 0.0
	if 'prior' in kwargs:
		prior = kwargs.pop('prior')
	if 'cost' in kwargs:
		cost = kwargs.pop('cost')
	if(len(kwargs) > 0):
		raise TypeError(list(kwargs.keys())[0] + ' is an invalid keyword argument')

	val = np.zeros(win_max*fail_max*button_max*time_max, dtype=np.float64)
	finite_arm_reward_table_cpp(val, win_max, fail_max, time_max, button_max, cost, prior[0], prior[1])
	val = val.reshape((time_max, button_max, win_max, fail_max))
	return val


#Finite arm transition table function
finite_arm_transition_table_cpp = lib.finite_arm_transition_table
finite_arm_transition_table_cpp.argtypes = [ctl.ndpointer(np.int32, flags='aligned, c_contiguous'),
											ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
											ctypes.c_double, ctypes.c_double, ctypes.c_double]

def finite_arm_transition_table(win_max, fail_max, time_max, button_max, **kwargs):
	prior = (1.0, 1.0)
	cost = 0.0
	if 'prior' in kwargs:
		prior = kwargs.pop('prior')
	if 'cost' in kwargs:
		cost = kwargs.pop('cost')
	if(len(kwargs) > 0):
		raise TypeError(list(kwargs.keys())[0] + ' is an invalid keyword argument')

	val = np.zeros(win_max*fail_max*button_max*time_max, dtype=np.int32)
	finite_arm_transition_table_cpp(val, win_max, fail_max, time_max, button_max, cost, prior[0], prior[1])
	val = val.reshape((time_max, button_max, win_max, fail_max))
	return val