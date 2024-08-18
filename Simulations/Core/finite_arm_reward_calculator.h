#include <cstdio>
#include <cmath>
#include <iostream>

#ifndef FIN_ARM_REWARD_CALCULATOR_H
#define FIN_ARM_REWARD_CALCULATOR_H

// Calculates optimal reward table using general recursive transition matrix calculator
class Probabilistic_Model_Finite{
public:
	Probabilistic_Model_Finite(): move_cost(0.0) {};
	Probabilistic_Model_Finite(double cost): move_cost(cost) {};

	const double move_cost;

	virtual double mean_step_reward(int win, int fail) = 0;
	virtual double success_transition_p(int win, int fail) = 0;
	virtual double expected_reward_from_last_button(int remaining) = 0;
};

class Finite_Arm_Reward_Table: public Probabilistic_Model_Finite{
private:
	int win_max;
	int fail_max;
	int time_max;
	int button_max;

	const bool extern_reward_table;
	double *reward_table;
	double *iter_table;

	void iter_table_prefill(){
		double next_button_val = mean_step_reward(0, 0) - move_cost;
		double stay_val;

		int col_space = win_max+time_max;
		int row_space = fail_max+time_max;
		for(int btn = 0; btn < button_max; ++btn){
			for(int i = 0; i < col_space; ++i){
				for(int j = 0; j < row_space; ++j){
					stay_val = mean_step_reward(i, j);
					iter_table[(btn*col_space+i)*row_space+j] = (stay_val > next_button_val ? stay_val : next_button_val);
				}
			}
		}
	}

	void iteration(int scope_win, int scope_fail, int remain){
		int row_space = fail_max+time_max;
		for(int btn = button_max-1; btn >= 0; --btn){
			double success_p = success_transition_p(0, 0);
			double next_button_val, stay_val;
			int shift;
			if(btn){
				shift = (win_max+time_max)*row_space*(btn-1);
				next_button_val = mean_step_reward(0, 0) - move_cost + success_p*iter_table[shift+row_space] + (1.0 - success_p)*iter_table[shift+1];
			}
			else{
				next_button_val = expected_reward_from_last_button(remain) - move_cost;
			}
			shift = (win_max+time_max)*row_space*btn;

			for(int i = 0; i < scope_win; i++){
				for(int j = 0; j < scope_fail; j++){
					success_p = success_transition_p(i, j);
					stay_val = mean_step_reward(i, j) + success_p*iter_table[shift+(i+1)*row_space+j] + (1.0 - success_p)*iter_table[shift+i*row_space+j+1];
					iter_table[shift+i*row_space+j] = (stay_val > next_button_val ? stay_val : next_button_val);
				}
			}
		}
	}

	void copy_to_reward_table(int time){
		int row_space = fail_max+time_max;
		double *reward_table_pnt = reward_table + time*button_max*win_max*fail_max;
		for(int btn = 0; btn < button_max; btn++){
			int shift = (win_max+time_max)*row_space*btn;
			for(int i = 0; i < win_max; i++){
				for(int j = 0; j < fail_max; j++){
					*(reward_table_pnt++) = iter_table[shift+i*row_space+j];
				}
			}
		}
	}

protected:
	void fill_reward_table(void){
		iter_table_prefill();
		copy_to_reward_table(0);
		for(int i = 1; i < time_max; i++){
			iteration(win_max+time_max-i-1, fail_max+time_max-i-1, i+1);
			copy_to_reward_table(i);
		}
	}

public:
	Finite_Arm_Reward_Table(int win_init, int fail_init, int time_init, int buttons_init, double cost):
		Probabilistic_Model_Finite(cost), win_max(win_init), fail_max(fail_init), time_max(time_init),
		button_max(buttons_init), extern_reward_table(false)
	{
		reward_table = new double[time_max*button_max*win_max*fail_max];
		iter_table = new double[button_max*(win_max+time_max)*(fail_max+time_max)];
	}

	Finite_Arm_Reward_Table(double *reward_table_init, int win_init, int fail_init, int time_init, int buttons_init, double cost):
		Probabilistic_Model_Finite(cost), win_max(win_init), fail_max(fail_init), time_max(time_init), button_max(buttons_init),
		reward_table(reward_table_init), extern_reward_table(true)
	{
		iter_table = new double[button_max*(win_max+time_max)*(fail_max+time_max)];
	}

	~Finite_Arm_Reward_Table(){
		if(!extern_reward_table) delete [] reward_table;
		delete [] iter_table;
	}

	double operator()(int time, int button, int wins, int fails) const{
		return reward_table[fail_max*(win_max*(button_max*time + button) + wins) + fails];
	}
};

class Finite_Arm_Transition_Table: public Probabilistic_Model_Finite{
private:
	int win_max;
	int fail_max;
	int time_max;
	int button_max;

	double *iter_table;
	int *iter_transition;
	int *transition_table;
	const bool extern_transition_table;

	void iter_table_prefill(){
		double next_button_val = mean_step_reward(0, 0) - move_cost;
		double stay_val;

		bool if_switch;
		int col_space = win_max+time_max;
		int row_space = fail_max+time_max;
		for(int btn = 0; btn < button_max; ++btn){
			for(int i = 0; i < col_space; ++i){
				for(int j = 0; j < row_space; ++j){
					stay_val = mean_step_reward(i, j);
					if_switch = next_button_val > stay_val;
					iter_table[(btn*col_space+i)*row_space+j] = (if_switch ? next_button_val : stay_val);
					if(i < win_max && j < fail_max)
						iter_transition[(btn*win_max+i)*fail_max+j] = (std::fabs(stay_val - next_button_val) < 1.0e-10 ? 2 : if_switch);
				}
			}
		}
	}

	void iteration(int scope_win, int scope_fail, int remain){
		int row_space = fail_max+time_max;
		bool if_switch;
		for(int btn = button_max-1; btn >= 0; --btn){
			double success_p = success_transition_p(0, 0);
			double next_button_val, stay_val;
			int shift;
			if(btn){
				shift = (win_max+time_max)*row_space*(btn-1);
				next_button_val = mean_step_reward(0, 0) - move_cost + success_p*iter_table[shift+row_space] + (1.0 - success_p)*iter_table[shift+1];
			}
			else{
				next_button_val = expected_reward_from_last_button(remain) - move_cost;
			}
			shift = (win_max+time_max)*row_space*btn;

			for(int i = 0; i < scope_win; i++){
				for(int j = 0; j < scope_fail; j++){
					success_p = success_transition_p(i, j);
					stay_val = mean_step_reward(i, j) + success_p*iter_table[shift+(i+1)*row_space+j] + (1.0 - success_p)*iter_table[shift+i*row_space+j+1];
					if_switch = next_button_val > stay_val;
					iter_table[shift+i*row_space+j] = (if_switch ? next_button_val : stay_val);
					if(i < win_max && j < fail_max)
						iter_transition[(btn*win_max+i)*fail_max+j] = (std::fabs(stay_val - next_button_val) < 1.0e-10 ? 2 : if_switch);
				}
			}
		}
	}

	void copy_to_transition_table(int time){
		int *transition_table_pnt = transition_table + time*button_max*win_max*fail_max;
		for(int btn = 0; btn < button_max; btn++){
			int shift = win_max*fail_max*btn;
			for(int i = 0; i < win_max; i++){
				for(int j = 0; j < fail_max; j++){
					*(transition_table_pnt++) = iter_transition[shift+i*fail_max+j];
				}
			}
		}
	}

protected:
	void fill_transition_table(void){
		iter_table_prefill();
		copy_to_transition_table(0);
		for(int i = 1; i < time_max; i++){
			iteration(win_max+time_max-i-1, fail_max+time_max-i-1, i+1);
			copy_to_transition_table(i);
		}
	}

public:
	Finite_Arm_Transition_Table(int win_init, int fail_init, int time_init, int buttons_init, double cost):
		Probabilistic_Model_Finite(cost), win_max(win_init), fail_max(fail_init), time_max(time_init),
		button_max(buttons_init), extern_transition_table(false)
	{
		iter_table = new double[button_max*(win_max+time_max)*(fail_max+time_max)];
		iter_transition = new int[button_max*win_max*fail_max];
		transition_table = new int[time_max*button_max*win_max*fail_max];
	}

	Finite_Arm_Transition_Table(int *transition_table_init, int win_init, int fail_init, int time_init, int buttons_init, double cost):
		Probabilistic_Model_Finite(cost), win_max(win_init), fail_max(fail_init), time_max(time_init), button_max(buttons_init),
		transition_table(transition_table_init), extern_transition_table(true)
	{
		iter_table = new double[button_max*(win_max+time_max)*(fail_max+time_max)];
		iter_transition = new int[button_max*win_max*fail_max];
	}

	~Finite_Arm_Transition_Table(){
		if(!extern_transition_table) delete [] transition_table;
		delete [] iter_table;
		delete [] iter_transition;
	}

	int operator()(int time, int button, int wins, int fails) const{
		return transition_table[fail_max*(win_max*(button_max*time + button) + wins) + fails];
	}
};

#endif