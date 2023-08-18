#include <trial.h>
#include <reward_calculator.h>
#include <transition_calculator.h>

#include <boost/math/special_functions/beta.hpp>

#ifndef TRUNCATED_BERNOULLI_H
#define TRUNCATED_BERNOULLI_H

//Does everything that bernoulli button does but truncates probability at trunc_high from above

class Truncated_Bernoulli_Button: public Button{
private:
	//Random number generator variables
	std::random_device rand_dev;	//Will be used to obtain a seed for the random number engine
	std::mt19937 generator; //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> distr;

	double trunc_high;

	double bernoulli_p;
	double reward;

public:
	void new_button(){
		bernoulli_p = trunc_high*distr(generator);
	}

	double print_probability(){
		std::cout << "Warning! Bernoulli button probability inquiry\n";
		return bernoulli_p;
	}

	double press_current_button(){
		double thrw = distr(generator);
		if(thrw < bernoulli_p) return reward;
		else return 0.0;
	}

	Truncated_Bernoulli_Button(double trunc_high_init = 0.9, double reward_init = 1.0): 
	reward(reward_init), trunc_high(trunc_high_init), rand_dev(), generator(rand_dev()), distr(0.0, 1.0)
	{
		new_button();
	}
};

class Truncated_Bernoulli_Reward: public Reward_Table{
	double trunc_high;

	double beta_inc(double alpha, double beta){
		return boost::math::beta <double, double, double> (alpha, beta, trunc_high);
	}

	double mean_step_reward(int win, int fail){
		return beta_inc(win+2, fail+1)/beta_inc(win+1, fail+1);
	}

	double success_transition_p(int win, int fail){
		return beta_inc(win+2, fail+1)/beta_inc(win+1, fail+1);
	}

public:
	Truncated_Bernoulli_Reward(int win_init, int fail_init, int time_init, double cost = 0.0, double trunc_high_0 = 0.9):
		Reward_Table(win_init, fail_init, time_init, cost), trunc_high(trunc_high_0)
		{
			fill_reward_table();
		};

	Truncated_Bernoulli_Reward(double *reward_table_init, int win_init, int fail_init, int time_init, double cost = 0.0, double trunc_high_0 = 0.9):
		Reward_Table(reward_table_init, win_init, fail_init, time_init, cost), trunc_high(trunc_high_0)
		{
			fill_reward_table();
		};
};

class Truncated_Bernoulli_Transition: public Transition_Table{
	double trunc_high;
	
	double beta_inc(double alpha, double beta){
		return boost::math::beta <double, double, double> (alpha, beta, trunc_high);
	}

	double mean_step_reward(int win, int fail){
		return beta_inc(win+2, fail+1)/beta_inc(win+1, fail+1);
	}

	double success_transition_p(int win, int fail){
		return beta_inc(win+2, fail+1)/beta_inc(win+1, fail+1);
	}

public:
	Truncated_Bernoulli_Transition(int win_init, int fail_init, int time_init, double cost, double trunc_high_0 = 0.9):
	Transition_Table(win_init, fail_init, time_init, cost), trunc_high(trunc_high_0)
	{
		fill_transition_table();
	};

	Truncated_Bernoulli_Transition(int *transition_table_init, int win_init, int fail_init, int time_init, double cost, double trunc_high_0 = 0.9):
	Transition_Table(transition_table_init, win_init, fail_init, time_init, cost), trunc_high(trunc_high_0)
	{
		fill_transition_table();
	};
};

// Implements Bernoulli policy decision making by referencing optimal transition table

class Truncated_Bernoulli_Policy: public Policy{
private:
	double trunc_high;

	int wins;
	int fails;
	int time;
	int time_left;

	Truncated_Bernoulli_Transition trans;

public:
	Truncated_Bernoulli_Policy(int time_init, double cost, double trunc_high_0 = 0.9):
		wins(0), fails(0), trunc_high(trunc_high_0),
		time(time_init), time_left(time_init), trans(time, time, time, cost, trunc_high_0) {};
	
	void reset(){time_left = time; wins = 0; fails = 0;}

	bool change_button(double reward){
		if(reward > 0.0) wins++;
		else fails++;
		time_left--;

		if(time_left == 0) return false;
		
		int trans_time = trans(wins, fails);

		if(trans_time > 0 && time_left >= trans_time){
			wins = fails = 0;
			return true;
		}
		else return false;
	}
};

#endif