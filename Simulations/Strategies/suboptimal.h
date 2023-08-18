#include <trial.h>

#ifndef SUBOPTIMAL_H
#define SUBOPTIMAL_H

class Random_Policy: public Policy{
private:
	std::random_device rand_dev;	//Will be used to obtain a seed for the random number engine
	std::mt19937 generator; //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> distr;

	double switch_p;

	int wins;
	int fails;
public:
	Random_Policy(double switch_p_i): switch_p(switch_p_i), rand_dev(), generator(rand_dev()), distr(0.0, 1.0) {};
	
	void reset(){return;}

	bool change_button(double reward){
		if(reward > 0.0) wins++;
		else fails++;

		double thrw = distr(generator);
		
		if(thrw < switch_p){
			wins = fails = 0;
			return true;
		}
		else return false;
	}
};

class Difference_Policy: public Policy{
private:
	int diff;

	int wins;
	int fails;
public:
	Difference_Policy(int diffi): diff(diffi) {};
	
	void reset(){wins = 0; fails = 0;}

	bool change_button(double reward){
		if(reward > 0.0) wins++;
		else fails++;
	
		if(fails > 0 && wins < fails+diff){
			wins = fails = 0;
			return true;
		}
		else return false;
	}
};

class Ratio_Policy: public Policy{
private:
	double ratio;

	int wins;
	int fails;
public:
	Ratio_Policy(double ratioi): ratio(ratioi) {};
	
	void reset(){wins = 0; fails = 0;}

	bool change_button(double reward){
		if(reward > 0.0) wins++;
		else fails++;
	
		if(fails > 0 && wins/(1.0*(wins+fails)) < ratio){
			wins = fails = 0;
			return true;
		}
		else return false;
	}
};

#endif