#include <iostream>
#include <cstring>
#include <timer.h>
#include <random>

#include <reward_calculator.h>
#include <transition_calculator.h>

#include <trial.h>
#include <bernoulli.h>
#include <meta_bernoulli.h>

void run_coefficient_table_generator(int argc, char const *argv[]){
	int win_max = 10, fail_max = 10, time_max = 100;
	double cost = 0.0;
	double alpha_prior = 1.0, beta_prior = 1.0;

	for(int i = 1; i < argc; i++){
		if(strcmp(argv[i], "-wins") == 0){
			win_max = atoi(argv[i+1]);
			i++;
		}
		if(strcmp(argv[i], "-fails") == 0){
			fail_max = atoi(argv[i+1]);
			i++;
		}
		if(strcmp(argv[i], "-time") == 0){
			time_max = atoi(argv[i+1]);
			i++;
		}
		if(strcmp(argv[i], "-cost") == 0){
			cost = atof(argv[i+1]);
			i++;
		}
		if(strcmp(argv[i], "-alpha") == 0){
			alpha_prior = atof(argv[i+1]);
			i++;
		}
		if(strcmp(argv[i], "-beta") == 0){
			beta_prior = atof(argv[i+1]);
			i++;
		}
	}

	Timer timer;
	Bernoulli_Reward coef(win_max, fail_max, time_max, cost, alpha_prior, beta_prior);
	timer.record();
	std::cout << "The calculation took " << timer.total << std::endl;
	std::cout << std::fixed;
	
	for(int i = 0; i < win_max; i++){
		for(int j = 0; j < fail_max; j++){
			std::cout << coef(time_max-1, i, j) << " ";
		}
		std::cout << std::endl;
	}

	return;
}

void run_transition_table_generator(int argc, char const *argv[]){
	int win_max = 10, fail_max = 10, time_max = 100;
	double cost = 0.0;
	double alpha_prior = 1.0, beta_prior = 1.0;

	for(int i = 1; i < argc; i++){
		if(strcmp(argv[i], "-wins") == 0){
			win_max = atoi(argv[i+1]);
			i++;
		}
		if(strcmp(argv[i], "-fails") == 0){
			fail_max = atoi(argv[i+1]);
			i++;
		}
		if(strcmp(argv[i], "-time") == 0){
			time_max = atoi(argv[i+1]);
			i++;
		}
		if(strcmp(argv[i], "-cost") == 0){
			cost = atof(argv[i+1]);
			i++;
		}
		if(strcmp(argv[i], "-alpha") == 0){
			alpha_prior = atof(argv[i+1]);
			i++;
		}
		if(strcmp(argv[i], "-beta") == 0){
			beta_prior = atof(argv[i+1]);
			i++;
		}
	}

	Timer timer;
	Bernoulli_Transition trans(win_max, fail_max, time_max, cost, alpha_prior, beta_prior);
	timer.record();
	std::cout << "The calculation took " << timer.total << std::endl;
	std::cout << std::fixed;
	
	for(int i = 0; i < win_max; i++){
		for(int j = 0; j < fail_max; j++){
			std::cout << trans(i, j) << " ";
		}
		std::cout << std::endl;
	}

	return;
}

void run_bernoulli_trials(int argc, char const *argv[]){
	int time = 100, num_tries = 100;
	double cost = 0.0;

	for(int i = 1; i < argc; i++){
		if(strcmp(argv[i], "-time") == 0){
			time = atoi(argv[i+1]);
			i++;
		}
		if(strcmp(argv[i], "-cost") == 0){
			cost = atof(argv[i+1]);
			i++;
		}
		if(strcmp(argv[i], "-tries") == 0){
			num_tries = atoi(argv[i+1]);
			i++;
		}
	}

	Bernoulli_Button bb;

	// for(int tr = 0; tr < 10; tr++){
	// 	int success = 0;

	// 	if(bb.press_next_button() > 0.0) success++;
	// 	for (int i = 1; i < 1000; i++)
	// 		if(bb.press_current_button() > 0.0) success++;

	// 	double true_p = bb.print_probability();
	// 	std::cout << "Estimated probability " << success/1000.0 << '\n';
	// 	std::cout << "True probability " << true_p << "\n\n";
	// }

	Bernoulli_Policy_Direct plc(time, cost);
	Policy_Test plc_test(&bb, &plc, time, cost, num_tries);
	Bernoulli_Reward coef(1, 1, time, cost);

	plc_test.run();
	// std::cout << "Theoretical expected reward: " << plc.table(time-1, 0, 0) << std::endl;
	std::cout << "Theoretical expected reward: " << coef(time-1, 0, 0) << std::endl;
	std::cout << "Experimental mean reward: " << plc_test.rewards_mean() << std::endl;
	std::cout << "Experimental reward deviation: " << plc_test.rewards_std() << std::endl;
	std::cout << "Experimental reward mean deviation: " << plc_test.rewards_std()/std::sqrt(1.0*num_tries) << std::endl;

	// plc.comments_on();
	// plc_test.single_run();
	// std::cout << plc_test.rewards_mean() << std::endl;

	return;
}

void run_biased_bernoulli_trials(int argc, char const *argv[]){
	int time = 100, num_tries = 100;
	double cost = 0.0;
	double alpha_prior = 1.0, beta_prior = 1.0;

	for(int i = 1; i < argc; i++){
		if(strcmp(argv[i], "-time") == 0){
			time = atoi(argv[i+1]);
			i++;
		}
		if(strcmp(argv[i], "-tries") == 0){
			num_tries = atoi(argv[i+1]);
			i++;
		}
		if(strcmp(argv[i], "-cost") == 0){
			cost = atof(argv[i+1]);
			i++;
		}
		if(strcmp(argv[i], "-alpha") == 0){
			alpha_prior = atof(argv[i+1]);
			i++;
		}
		if(strcmp(argv[i], "-beta") == 0){
			beta_prior = atof(argv[i+1]);
			i++;
		}
	}

	Biased_Bernoulli_Button bbb(alpha_prior, beta_prior);

	Bernoulli_Policy_Direct plc(time, cost, alpha_prior, beta_prior);
	Policy_Test plc_test(&bbb, &plc, time, cost, num_tries);
	Bernoulli_Reward coef(1, 1, time, cost);

	plc_test.run();
	// std::cout << "Theoretical expected reward: " << plc.table(time-1, 0, 0) << std::endl;
	std::cout << "Theoretical expected reward: " << coef(time-1, 0, 0) << std::endl;
	std::cout << "Experimental mean reward: " << plc_test.rewards_mean() << std::endl;
	std::cout << "Experimental reward deviation: " << plc_test.rewards_std() << std::endl;
	std::cout << "Experimental reward mean deviation: " << plc_test.rewards_std()/std::sqrt(1.0*num_tries) << std::endl;

	return;
}

void run_meta_bernoulli_trials(int argc, char const *argv[]){
	int time = 100, num_tries = 100;
	double cost = 0.0;
	double alpha_prior = 1.0, beta_prior = 1.0;
	double prior_k = 1.0;

	for(int i = 1; i < argc; i++){
		if(strcmp(argv[i], "-time") == 0){
			time = atoi(argv[i+1]);
			i++;
		}
		if(strcmp(argv[i], "-tries") == 0){
			num_tries = atoi(argv[i+1]);
			i++;
		}
		if(strcmp(argv[i], "-cost") == 0){
			cost = atof(argv[i+1]);
			i++;
		}
		if(strcmp(argv[i], "-alpha") == 0){
			alpha_prior = atof(argv[i+1]);
			i++;
		}
		if(strcmp(argv[i], "-beta") == 0){
			beta_prior = atof(argv[i+1]);
			i++;
		}
		if(strcmp(argv[i], "-k") == 0){
			prior_k = atof(argv[i+1]);
			i++;
		}
	}

	Biased_Bernoulli_Button bbb(alpha_prior, beta_prior);

	Bernoulli_Policy_Direct plc_ub(time, cost);
	Meta_Bernoulli_Policy plc_meta(time, prior_k);

	Policy_Test plc_test_ub(&bbb, &plc_ub, time, cost, num_tries);
	Policy_Test plc_test_meta(&bbb, &plc_meta, time, cost, num_tries);

	Bernoulli_Reward coef(1, 1, time, cost, alpha_prior, beta_prior);

	Timer timer;
	plc_test_ub.run();
	timer.record();
	std::cout << "The unbiased calculation took " << timer.last_period << std::endl;

	plc_test_meta.run();
	timer.record();
	std::cout << "The meta calculation took " << timer.last_period << "\n" << std::endl;

	// std::cout << "Theoretical expected reward: " << plc.table(time-1, 0, 0) << std::endl;
	std::cout << "Theoretical expected reward: " << coef(time-1, 0, 0) << std::endl;
	std::cout << "\nUnbiased run\n\n";
	std::cout << "Experimental mean reward: " << plc_test_ub.rewards_mean() << std::endl;
	std::cout << "Experimental reward deviation: " << plc_test_ub.rewards_std() << std::endl;
	std::cout << "Experimental reward mean deviation: " << plc_test_ub.rewards_std()/std::sqrt(1.0*num_tries) << std::endl;
	std::cout << "\nDistribution learning run\n\n";
	std::cout << "Experimental mean reward: " << plc_test_meta.rewards_mean() << std::endl;
	std::cout << "Experimental reward deviation: " << plc_test_meta.rewards_std() << std::endl;
	std::cout << "Experimental reward mean deviation: " << plc_test_meta.rewards_std()/std::sqrt(1.0*num_tries) << std::endl;

	return;
}

int main(int argc, char const *argv[]){
	// run_coefficient_table_generator(argc, argv);
	// run_bernoulli_trials(argc, argv);
	// run_biased_bernoulli_trials(argc, argv);
	// run_transition_table_generator(argc, argv);
	run_meta_bernoulli_trials(argc, argv);

	return 0.0;
}