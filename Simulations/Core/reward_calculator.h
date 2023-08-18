#include <cstdio>
#include <cmath>

#ifndef COEF_CALCULATOR_H
#define COEF_CALCULATOR_H

//Defines the model to be recursively solved via dynamic programming
#ifndef PROBABILISTIC_MODEL_CLASS
#define PROBABILISTIC_MODEL_CLASS

class Probabilistic_Model{
public:
	Probabilistic_Model(): move_cost(0.0) {};
	Probabilistic_Model(double cost): move_cost(cost) {};

	const double move_cost;

	virtual double mean_step_reward(int win, int fail) = 0;
	virtual double success_transition_p(int win, int fail) = 0;
};

#endif

// Calculates optimal reward table using general recursive transition matrix calculator

class Reward_Table: public Probabilistic_Model{
private:
	int win_max;
	int fail_max;
	int time_max;

	const bool extern_reward_table;
	double *reward_table;
	double *iter_table;

	void iter_table_prefill(){
		double next_button_val = mean_step_reward(0, 0) - move_cost;
		double stay_val;

		int row_space = fail_max+time_max;
		for(int i = 0; i < (win_max+time_max); i++){
			for(int j = 0; j < row_space; j++){
				stay_val = mean_step_reward(i, j);
				
				iter_table[i*row_space+j] =(stay_val > next_button_val ? stay_val : next_button_val);
			}
		}
	}

	void iteration(int scope_win, int scope_fail){
		int row_space = fail_max+time_max;
		double success_p = success_transition_p(0, 0);
		double next_button_val = mean_step_reward(0, 0) - move_cost + success_p*iter_table[row_space] + (1.0 - success_p)*iter_table[1];
		double stay_val;

		for(int i = 0; i < scope_win; i++){
			for(int j = 0; j < scope_fail; j++){
				success_p = success_transition_p(i, j);
				stay_val = mean_step_reward(i, j) + success_p*iter_table[(i+1)*row_space+j] + (1.0 - success_p)*iter_table[i*row_space+j+1];
				iter_table[i*row_space+j] = (stay_val > next_button_val ? stay_val : next_button_val);
			}
		}
	}

	void copy_to_reward_table(int time){
		int row_space = fail_max+time_max;
		double *reward_table_pnt = reward_table + time*win_max*fail_max;
		for(int i = 0; i < win_max; i++){
			for(int j = 0; j < fail_max; j++){
				*(reward_table_pnt++) = iter_table[i*row_space+j];
			}
		}
	}

protected:
	void fill_reward_table(void){
		iter_table_prefill();
		copy_to_reward_table(0);
		for(int i = 1; i < time_max; i++){
			iteration(win_max+time_max-i-1, fail_max+time_max-i-1);
			copy_to_reward_table(i);
		}
	}

public:
	Reward_Table(int win_init, int fail_init, int time_init, double cost):
		Probabilistic_Model(cost), win_max(win_init), fail_max(fail_init), time_max(time_init), extern_reward_table(false)
	{
		reward_table = new double[time_max*win_max*fail_max];
		iter_table = new double[(win_max+time_max)*(fail_max+time_max)];
	}

	Reward_Table(double *reward_table_init, int win_init, int fail_init, int time_init, double cost):
		Probabilistic_Model(cost), win_max(win_init), fail_max(fail_init), time_max(time_init), reward_table(reward_table_init), extern_reward_table(true)
	{
		iter_table = new double[(win_max+time_max)*(fail_max+time_max)];
	}

	~Reward_Table(){
		if(!extern_reward_table) delete [] reward_table;
		delete [] iter_table;
	}

	double operator()(int time, int alpha, int beta) const{
		return reward_table[fail_max*(win_max*time + alpha) + beta];
	}
};

#endif