#include <iostream>
#include <cstring>
#include <timer.h>
#include <random>

#include <reward_calculator.h>
#include <transition_calculator.h>
#include <finite_arm_reward_calculator.h>

#include <trial.h>
#include <bernoulli.h>
#include <meta_bernoulli.h>
#include <truncated_bernoulli.h>
#include <suboptimal.h>
#include <finite_bernoulli.h>

// This function is used when expected_reward_table is called in Python
extern "C" void reward_table(double *table, int win_max, int fail_max, int time_max, double cost, double alpha_prior, double beta_prior)
{
	Bernoulli_Reward coef(table, win_max, fail_max, time_max, cost, alpha_prior, beta_prior);

	return;
}

// This function is used when expected_reward_table is called in Python
extern "C" void transition_table(int *table, int win_max, int fail_max, int time_max, double cost, double alpha_prior, double beta_prior)
{
	Bernoulli_Transition trans(table, win_max, fail_max, time_max, cost, alpha_prior, beta_prior);

	return;
}

// These functions implement the same routines as above for Bernoulli trials with probability truncated from the above

extern "C" void truncated_reward_table(double *table, int win_max, int fail_max, int time_max, double cost, double trunc_high)
{
	Truncated_Bernoulli_Reward coef(table, win_max, fail_max, time_max, cost, trunc_high);

	return;
}

extern "C" void truncated_transition_table(int *table, int win_max, int fail_max, int time_max, double cost, double trunc_high)
{
	Truncated_Bernoulli_Transition trans(table, win_max, fail_max, time_max, cost, trunc_high);

	return;
}

// Routine that runs arbitraty policy with arbitrary button

inline Button* button_choice(char button_type, double *button_param){
	Button *btn;

	//Bernoulli button, button_a_prior = button_param[0], button_b_prior = button_param[1]
	if(button_type == 'b') btn = new Biased_Bernoulli_Button(button_param[0], button_param[1]);
	//Truncated Bernoulli button, trunc_button = button_param[0]
	else if(button_type == 't') btn = new Truncated_Bernoulli_Button(button_param[0]);

	return btn;
}

inline Policy* policy_choice(int time, char policy_type, double *policy_param){
	Policy *plc;

	//Bernoulli button policy, policy_cost = policy_param[0], policy_a_prior = policy_param[1], policy_b_prior = policy_param[2]
	if(policy_type == 'b') plc = new Bernoulli_Policy_Direct(time, policy_param[0], policy_param[1], policy_param[2]);
	//Bernoulli button policy with metalearning, prior_k = policy_param[0]
	else if(policy_type == 'm') plc = new Meta_Bernoulli_Policy(time, policy_param[0]);
	//Bernoulli button policy with truncated probability, policy_cost = policy_param[0], trunc_policy = policy_param[1]
	else if(policy_type == 't') plc = new Truncated_Bernoulli_Policy(time, policy_param[0], policy_param[1]);
	//Random button policy, switch_p = policy_param[0]
	else if(policy_type == 'r') plc = new Random_Policy(policy_param[0]);
	//Difference button policy, diff = policy_param[0]
	else if(policy_type == 'd') plc = new Difference_Policy(policy_param[0]);
	//Ratio button policy, ratio = policy_param[0]
	else if(policy_type == 'f') plc = new Ratio_Policy(policy_param[0]);
	//Bernoulli finite button policy, nbuttons = policy_param[0], policy_cost = policy_param[1], policy_a_prior = policy_param[2], policy_b_prior = policy_param[3]
	else if(policy_type == 'n') plc = new Finite_Bernoulli_Policy(time, policy_param[0], policy_param[1], policy_param[2], policy_param[3]);

	return plc;
}

extern "C" void run_trials(double *rewards, int time, double cost, int num_tries, char button_type, double *button_param, char policy_type, double *policy_param)
{
	Button *btn = button_choice(button_type, button_param);
	Policy *plc = policy_choice(time, policy_type, policy_param);

	Policy_Test plc_run(btn, plc, time, cost, num_tries);

	for(int i = 0; i < num_tries; i++){
		rewards[i] = plc_run.single_run();
	}
}

extern "C" void record_full_trials(double *rewards, int time, double cost, int num_tries, char button_type, double *button_param, char policy_type, double *policy_param)
{
	double *rewards_tmp = rewards;

	Button *btn = button_choice(button_type, button_param);
	Policy *plc = policy_choice(time, policy_type, policy_param);

	Single_Policy_Run plc_run(btn, plc, time, cost);

	for(int i = 0; i < num_tries; i++){
		plc_run.single_run(rewards_tmp);
		rewards_tmp += 4*time;
	}
}

extern "C" void finite_arm_reward_table(double *table, int win_max, int fail_max, int time_max, int button_max, double cost, double alpha_prior, double beta_prior)
{
	Finite_Arm_Bernoulli_Reward coef(table, win_max, fail_max, time_max, button_max, cost, alpha_prior, beta_prior);

	return;
}

extern "C" void finite_arm_transition_table(int *table, int win_max, int fail_max, int time_max, int button_max, double cost, double alpha_prior, double beta_prior)
{
	Finite_Arm_Bernoulli_Transition coef(table, win_max, fail_max, time_max, button_max, cost, alpha_prior, beta_prior);

	return;
}