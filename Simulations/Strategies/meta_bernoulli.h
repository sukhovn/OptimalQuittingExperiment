#include <trial.h>
#include <inference_bernoulli.h>
#include <transition_calculator.h>
#include <iostream>

#ifndef META_BERNOULLI_H
#define META_BERNOULLI_H

class Bernoulli_Transition_Threshold: public Transition_Value{
	double alpha_prior;
	double beta_prior;

	double mean_step_reward(int win, int fail){
		return (alpha_prior+win)/(win+fail+alpha_prior+beta_prior);
	}

	double success_transition_p(int win, int fail){
		return (alpha_prior+win)/(win+fail+alpha_prior+beta_prior);
	}

public:
	Bernoulli_Transition_Threshold(): Transition_Value(0.0) {};

	int check(int win, int fail, int time_left, double alpha_0, double beta_0){
		alpha_prior = alpha_0; beta_prior = beta_0;
		return Transition_Value::check(win, fail, time_left);
	}
};

class Meta_Bernoulli_Policy: public Policy{
private:
	int wins;
	int fails;
	int time;
	int time_left;

	Bernoulli_Hierarchy bh;
	Bernoulli_Transition_Threshold tc;

public:
	Meta_Bernoulli_Policy(int time_init, double prior_k = 1.0): wins(0), fails(0), time(time_init), time_left(time_init), bh(prior_k), tc() {};
	
	void reset(){time_left = time; wins = 0; fails = 0; bh.clear();}

	bool change_button(double reward){
		if(reward > 0.0){
			bh.add_win();
			wins++;
		}
		else{
			bh.add_fail();
			fails++;
		}
		time_left--;

		if(time_left == 0) return false;
		
		bh.find_map();
		// std::cout << "Remaining presses: " << time_left << std::endl;
		// std::cout << "Inferred alpha and beta: " << bh.alpha_map << ", " << bh.beta_map << std::endl;
		int trans_time = tc.check(wins, fails, time_left, bh.alpha_map, bh.beta_map);
		// std::cout << "Estimated transition time: " << trans_time << std::endl;

		if(trans_time > 0 && time_left >= trans_time){
			wins = fails = 0;
			bh.add_button(0, 0);
			return true;
		}
		else return false;
	}
};

#endif