#include <trial.h>
#include <finite_arm_reward_calculator.h>
#include <bernoulli.h>

#ifndef FINITE_BERNOULLI_H
#define FINITE_BERNOULLI_H

// Calculates Bernoulli coefficient table using general recursive transition table calculator for finite number of buttons

class Finite_Arm_Bernoulli_Reward: public Finite_Arm_Reward_Table{
	double alpha_prior;
	double beta_prior;

	double mean_step_reward(int win, int fail){
		return (alpha_prior+win)/(win+fail+alpha_prior+beta_prior);
	}

	double success_transition_p(int win, int fail){
		return (alpha_prior+win)/(win+fail+alpha_prior+beta_prior);
	}

	double expected_reward_from_last_button(int remaining){
		return remaining*alpha_prior/(alpha_prior+beta_prior);
	}

public:
	Finite_Arm_Bernoulli_Reward(int win_init, int fail_init, int time_init, int buttons_init, double cost, double alpha_0 = 1.0, double beta_0 = 1.0):
		Finite_Arm_Reward_Table(win_init, fail_init, time_init, buttons_init, cost), alpha_prior(alpha_0), beta_prior(beta_0)
	{
		fill_reward_table();
	};

	Finite_Arm_Bernoulli_Reward(double *reward_table_init, int win_init, int fail_init, int time_init, int buttons_init, double cost, double alpha_0 = 1.0, double beta_0 = 1.0):
		Finite_Arm_Reward_Table(reward_table_init, win_init, fail_init, time_init, buttons_init, cost), alpha_prior(alpha_0), beta_prior(beta_0)
	{
		fill_reward_table();
	};
};

// Calculates Bernoulli transition table using general recursive transition table calculator for finite number of buttons

class Finite_Arm_Bernoulli_Transition: public Finite_Arm_Transition_Table{
	double alpha_prior;
	double beta_prior;

	double mean_step_reward(int win, int fail){
		return (alpha_prior+win)/(win+fail+alpha_prior+beta_prior);
	}

	double success_transition_p(int win, int fail){
		return (alpha_prior+win)/(win+fail+alpha_prior+beta_prior);
	}

	double expected_reward_from_last_button(int remaining){
		return remaining*alpha_prior/(alpha_prior+beta_prior);
	}

public:
	Finite_Arm_Bernoulli_Transition(int win_init, int fail_init, int time_init, int buttons_init, double cost, double alpha_0 = 1.0, double beta_0 = 1.0):
		Finite_Arm_Transition_Table(win_init, fail_init, time_init, buttons_init, cost), alpha_prior(alpha_0), beta_prior(beta_0)
	{
		fill_transition_table();
	};

	Finite_Arm_Bernoulli_Transition(int *transition_table_init, int win_init, int fail_init, int time_init, int buttons_init, double cost, double alpha_0 = 1.0, double beta_0 = 1.0):
		Finite_Arm_Transition_Table(transition_table_init, win_init, fail_init, time_init, buttons_init, cost), alpha_prior(alpha_0), beta_prior(beta_0)
	{
		fill_transition_table();
	};
};

class Finite_Bernoulli_Policy: public Policy{
private:
	double alpha_prior;
	double beta_prior;
	int wins;
	int fails;
	int time;
	int time_left;
	int buttons;
	int buttons_left;

	Finite_Arm_Bernoulli_Transition trans;

public:
	Finite_Bernoulli_Policy(int time_init, int buttons_init, double cost, double alpha_0 = 1.0, double beta_0 = 1.0):
		wins(0), fails(0), alpha_prior(alpha_0), beta_prior(beta_0), time(time_init), time_left(time_init), 
		buttons(buttons_init), buttons_left(buttons_init-1), trans(time-1, time-1, time-1, buttons_init-1, cost, alpha_0, beta_0) {};
	
	void reset(){time_left = time; buttons_left = buttons-1; wins = 0; fails = 0;}

	bool change_button(double reward){
		if(reward > 0.0) wins++;
		else fails++;
		time_left--;

		if(time_left == 0 || buttons_left == 0) return false;

		if(trans(time_left-1, buttons_left-1, wins, fails) == 1){
			wins = fails = 0;
			--buttons_left;
			return true;
		}
		else return false;
	}
};

#endif