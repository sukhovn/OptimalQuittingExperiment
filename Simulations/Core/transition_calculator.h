#include <cstdio>
#include <cmath>
#include <iostream>

#ifndef TRANSITION_CALCULATOR_ALT_H
#define TRANSITION_CALCULATOR_ALT_H

//Calculates the optimal transition table
class Transition_Table: public Probabilistic_Model{
private:
	int win_max;
	int fail_max;
	int time_max;

	int time_curr;

	const bool extern_transition_table;
	int *transition_table;
	double *iter_table;

	void iter_table_prefill(){
		double next_button_val = mean_step_reward(0, 0) - move_cost;
		double stay_val;
		time_curr++;

		int row_space = fail_max+time_max;
		for(int i = 0; i < (win_max+time_max); i++){
			for(int j = 0; j < row_space; j++){
				stay_val = mean_step_reward(i, j);
				
				iter_table[i*row_space+j] = (stay_val > next_button_val ? stay_val : next_button_val);
			}
		}

		for(int i = 0; i < win_max; i++){
			for(int j = 0; j < fail_max; j++){
				if(iter_table[i*row_space+j] == next_button_val && transition_table[i*fail_max+j] == 0)
					transition_table[i*fail_max+j] = time_curr;
			}
		}			
	}

	void iteration(int scope_win, int scope_fail){
		int row_space = fail_max+time_max;
		double success_p = success_transition_p(0, 0);
		double next_button_val = mean_step_reward(0, 0) - move_cost + success_p*iter_table[row_space] + (1.0 - success_p)*iter_table[1];
		double stay_val;
		time_curr++;

		for(int i = 0; i < scope_win; i++){
			for(int j = 0; j < scope_fail; j++){
				success_p = success_transition_p(i, j);
				stay_val = mean_step_reward(i, j) + success_p*iter_table[(i+1)*row_space+j] + (1.0 - success_p)*iter_table[i*row_space+j+1];

				iter_table[i*row_space+j] = (stay_val > next_button_val ? stay_val : next_button_val);
			}
		}

		for(int i = 0; i < win_max; i++){
			for(int j = 0; j < fail_max; j++){
				if(transition_table[i*fail_max+j] != 0 && iter_table[i*row_space+j] > next_button_val){
					std::cout << "Warning! In case of " << i << " wins, " << j << " fails and " << move_cost << " button change cost\n";
					std::cout << "it becomes again better to stay with the current button rather than move to the next one" << std::endl;
				}
				if(iter_table[i*row_space+j] == next_button_val && transition_table[i*fail_max+j] == 0)
					transition_table[i*fail_max+j] = time_curr;
			}
		}
	}

protected:
	void fill_transition_table(void){
		iter_table_prefill();
		for(int i = 1; i < time_max; i++){
			iteration(win_max+time_max-i-1, fail_max+time_max-i-1);
			if(transition_table[(win_max-1)*fail_max+1] != 0) break;
		}
	}

public:
	Transition_Table(int win_init, int fail_init, int time_init, double cost):
		Probabilistic_Model(cost), win_max(win_init), fail_max(fail_init), time_max(time_init), time_curr(0), extern_transition_table(false)
	{
		iter_table = new double[(win_max+time_max)*(fail_max+time_max)];
		transition_table = new int[win_max*fail_max];

		for(int i = 0; i < win_max; i++)
			for(int j = 0; j < fail_max; j++)
				transition_table[i*fail_max+j] = 0;
	};

	Transition_Table(int *transition_table_init, int win_init, int fail_init, int time_init, double cost):
		Probabilistic_Model(cost), win_max(win_init), fail_max(fail_init), time_max(time_init), time_curr(0),
		transition_table(transition_table_init), extern_transition_table(true)
	{
		iter_table = new double[(win_max+time_max)*(fail_max+time_max)];

		for(int i = 0; i < win_max; i++)
			for(int j = 0; j < fail_max; j++)
				transition_table[i*fail_max+j] = 0;
	};

	~Transition_Table(){
		if(!extern_transition_table) delete [] transition_table;
		delete [] iter_table;
	}

	double operator()(int win, int fail) const{
		return transition_table[fail_max*win + fail];
	}
};

//Calculates optimal transition time for a given number of wins and fails on a fly
//This method does no precalculation so it is typically slow
class Transition_Value: public Probabilistic_Model{
private:
	int win_check;
	int fail_check;
	int time_max;

	int iter_table_size;
	double *iter_table;

	bool iter_table_prefill(){
		double next_button_val = mean_step_reward(0, 0) - move_cost;
		double stay_val;

		int row_space = fail_check+time_max;
		for(int i = 0; i < (win_check+time_max); i++){
			for(int j = 0; j < row_space; j++){
				stay_val = mean_step_reward(i, j);
				
				iter_table[i*row_space+j] = (stay_val > next_button_val ? stay_val : next_button_val);
			}
		}

		if(iter_table[win_check*row_space+fail_check] == next_button_val) return true;
		else return false;		
	}

	bool iteration(int scope_win, int scope_fail){
		int row_space = fail_check+time_max;
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

		if(iter_table[win_check*row_space+fail_check] == next_button_val) return true;
		else return false;
	}

public:
	Transition_Value(double cost): Probabilistic_Model(cost), win_check(0), fail_check(0), time_max(0), iter_table_size(0) {};

	~Transition_Value(){
		if(iter_table_size > 0) delete [] iter_table;
	}

	int check(int win, int fail, int time_left){
		win_check = win; fail_check = fail; time_max = time_left;

		int size_try = (win_check+time_max)*(fail_check+time_max);
		if(size_try > iter_table_size){
			if(iter_table_size > 0) delete [] iter_table;
			iter_table = new double[size_try];
			iter_table_size = size_try; 
		}

		if(iter_table_prefill()) return 1;
		for(int i = 1; i < time_max; i++){
			if(iteration(win_check+time_max-i-1, fail_check+time_max-i-1)) return i+1;
		}

		return 0;
	}
};

#endif