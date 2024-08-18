#include <trial.h>
#include <reward_calculator.h>
#include <transition_calculator.h>

#ifndef BERNOULLI_H
#define BERNOULLI_H

// std::random_device rand_dev;  //Will be used to obtain a seed for the random number engine
// std::mt19937 generator(rand_dev());; //Standard mersenne_twister_engine seeded with rd()
// std::uniform_real_distribution<> distr(0.0, 1.0);
 
// for (int i = 0; i < 10; ++i)
// 	std::cout << distr(generator) << '\n';

// Simulates Bernoulli button drawn out of the uniform distribution

class Bernoulli_Button: public Button{
private:
	//Random number generator variables
	std::random_device rand_dev;	//Will be used to obtain a seed for the random number engine
	std::mt19937 generator; //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> distr;

	double bernoulli_p;
	double reward;

public:
	void new_button(){
		bernoulli_p = distr(generator);
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

	Bernoulli_Button(double reward_init = 1.0): reward(reward_init), rand_dev(), generator(rand_dev()), distr(0.0, 1.0){
		new_button();
	}
};

// Simulates Bernoulii button drawn out of the Beta(\alpha, \beta) distribution

class Biased_Bernoulli_Button: public Button{
private:
	//Random number generator variables
	std::random_device rand_dev;	//Will be used to obtain a seed for the random number engine
	std::mt19937 generator; //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> uniform_distr;

	//Used to get bernoulli_p out of beta(alpha, beta) distribution
	//If X ~ Gamma(alpha, lambda), Y ~ Gamma(beta, lambda), then X/(X+Y) ~ Beta(alpha, beta)
	std::gamma_distribution<> gamma_alpha;
	std::gamma_distribution<> gamma_beta;
	
	double bernoulli_p;
	double reward;

public:
	const double alpha;
	const double beta;

	void new_button(){
		double ga_throw = gamma_alpha(generator);
		double gb_throw = gamma_beta(generator);
		bernoulli_p = ga_throw/(ga_throw+gb_throw);
	}

	double print_probability(){
		std::cout << "Warning! Bernoulli button probability inquiry\n";
		return bernoulli_p;
	}

	double press_current_button(){
		double thrw = uniform_distr(generator);
		if(thrw < bernoulli_p) return reward;
		else return 0.0;
	}

	Biased_Bernoulli_Button(double alpha_0 = 1.0, double beta_0 = 1.0, double reward_init = 1.0): reward(reward_init), rand_dev(), generator(rand_dev()), 
		uniform_distr(0.0, 1.0), alpha(alpha_0), beta(beta_0), gamma_alpha(alpha_0, 1.0), gamma_beta(beta_0, 1.0)
	{
		new_button();
	}
};

// Calculates Bernoulli coefficient table using general recursive transition table calculator

class Bernoulli_Reward: public Reward_Table{
	double alpha_prior;
	double beta_prior;

	double mean_step_reward(int win, int fail){
		return (alpha_prior+win)/(win+fail+alpha_prior+beta_prior);
	}

	double success_transition_p(int win, int fail){
		return (alpha_prior+win)/(win+fail+alpha_prior+beta_prior);
	}

public:
	Bernoulli_Reward(int win_init, int fail_init, int time_init, double cost, double alpha_0 = 1.0, double beta_0 = 1.0): Reward_Table(win_init, fail_init, time_init, cost),
																										alpha_prior(alpha_0), beta_prior(beta_0){
																											fill_reward_table();
																										};

	Bernoulli_Reward(double *reward_table_init, int win_init, int fail_init, int time_init, double cost, double alpha_0 = 1.0, double beta_0 = 1.0):
																										Reward_Table(reward_table_init, win_init, fail_init, time_init, cost),
																										alpha_prior(alpha_0), beta_prior(beta_0){
																											fill_reward_table();
																										};
};

// Calculates Bernoulli transition table using general recursive transition table calculator

class Bernoulli_Transition: public Transition_Table{
	double alpha_prior;
	double beta_prior;

	double mean_step_reward(int win, int fail){
		return (alpha_prior+win)/(win+fail+alpha_prior+beta_prior);
	}

	double success_transition_p(int win, int fail){
		return (alpha_prior+win)/(win+fail+alpha_prior+beta_prior);
	}

public:
	Bernoulli_Transition(int win_init, int fail_init, int time_init, double cost, double alpha_0 = 1.0, double beta_0 = 1.0): Transition_Table(win_init, fail_init, time_init, cost),
																										alpha_prior(alpha_0), beta_prior(beta_0){
																											fill_transition_table();
																										};

	Bernoulli_Transition(int *transition_table_init, int win_init, int fail_init, int time_init, double cost, double alpha_0 = 1.0, double beta_0 = 1.0): 
																										Transition_Table(transition_table_init, win_init, fail_init, time_init, cost),
																										alpha_prior(alpha_0), beta_prior(beta_0){
																											fill_transition_table();
																										};
};

// Implements Bernoulli policy decision making by referencing optimal transition table

class Bernoulli_Policy_Direct: public Policy{
private:
	double alpha_prior;
	double beta_prior;
	int wins;
	int fails;
	int time;
	int time_left;

	Bernoulli_Transition trans;

public:
	Bernoulli_Policy_Direct(int time_init, double cost, double alpha_0 = 1.0, double beta_0 = 1.0): wins(0), fails(0), alpha_prior(alpha_0), beta_prior(beta_0), 
		time(time_init), time_left(time_init), trans(time, time, time, cost, alpha_0, beta_0) {};
	
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