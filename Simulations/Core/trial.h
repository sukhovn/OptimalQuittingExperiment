#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

#ifndef TRIAL_H
#define TRIAL_H

class Button{
public:
	// Draws a new button
	virtual void new_button() = 0;

	// Presses the current button and outputs resulting reward
	virtual double press_current_button() = 0;

	double press_next_button()
	{
		new_button();
		return press_current_button();
	}
};

class Policy{
public:
	// Resets the policy memory tracking
	virtual void reset() = 0;

	// Outputs the decision whether the button should be changed based on the reward received
	virtual bool change_button(double reward) = 0;
};

class Policy_Test{
private:
	int time;
	int run_times;

	double move_cost;

public:
	Button *btn;
	Policy *pol;
	double *rewards;

	Policy_Test(Button *btn_init, Policy *pol_init, int time_init, double cost, int run_times_init):
		btn(btn_init), pol(pol_init), move_cost(cost), time(time_init), run_times(run_times_init){
			rewards = new double[run_times];
		};
	~Policy_Test(){
		delete [] rewards;
	}

	double single_run(){
		double reward_curr;
		double reward_acc = 0.0;
		bool change_button = false;

		btn->new_button();
		pol->reset();
		for(int i = 0; i < time; i++){
			if(change_button){
				reward_acc -= move_cost;
				reward_curr = btn->press_next_button();
			}
			else reward_curr = btn->press_current_button();
			change_button = pol->change_button(reward_curr);
			reward_acc += reward_curr;
		}

		return reward_acc;
	}

	void run(){
		for(int i = 0; i < run_times; i++) rewards[i] = single_run();
	}

	double rewards_mean(){
		double tmp = 0.0;
		for(int i = 0; i < run_times; i++) tmp += rewards[i]; 
		return tmp/(1.0*run_times);
	}

	double rewards_std(){
		double tmp = 0;
		double mean = rewards_mean();
		for(int i = 0; i < run_times; i++) tmp += std::pow(rewards[i] - mean, 2.0); 
		double nminus = (double) run_times-1;

		return std::sqrt(tmp/nminus);
	}
};


// Runs a button given a policy a single time

class Single_Policy_Run{
private:
	int time;

	double move_cost;

public:
	Button *btn;
	Policy *pol;

	Single_Policy_Run(Button *btn_init, Policy *pol_init, int time_init, double cost):
		btn(btn_init), pol(pol_init), move_cost(cost), time(time_init) {};

	double single_run(double *results){
		int indx = 0;
		int button_no = 0, wins = 0, fails = 0;
		double reward_curr;
		double reward_acc = 0.0;
		bool change_button = false;

		btn->new_button();
		pol->reset();
		for(int i = 0; i < time; i++){
			if(change_button){
				wins = 0;
				fails = 0;
				button_no++;
				reward_acc -= move_cost;
				reward_curr = btn->press_next_button();
			}
			else reward_curr = btn->press_current_button();

			if(reward_curr > 0.0) wins++;
			else fails++;

			change_button = pol->change_button(reward_curr);
			reward_acc += reward_curr;
			results[indx++] = (double) button_no;
			results[indx++] = (double) wins;
			results[indx++] = (double) fails;
			results[indx++] = reward_curr;
		}

		return reward_acc;
	}
};

#endif