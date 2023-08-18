#include <cmath>
#include <nelder-mead.h>

#ifndef INFERENCE_H
#define INFERENCE_H

//This function computes log(B(alpha + n, beta + m)/B(alpha, beta))
static double beta_frac_log(double alpha, double beta, int n, int m){
	double tmp = 1.0;
	for(int i = 0; i < n; i++) tmp *= (alpha+i);
	for(int i = 0; i < m; i++) tmp *= (beta+i);
	for(int i = 0; i < n+m; i++) tmp /= (alpha+beta+i);
	return std::log(tmp);
}

class Bernoulli_Hierarchy: private Nelder_Mead{
private:
	std::vector<std::vector<int>> buttons;

	double prior_k;
	double prior_theta;

	double log_prior(double alpha, double beta){
		return (prior_k-1)*std::log(alpha*beta) - (alpha+beta)/prior_theta;
	}

	double function(const std::vector<double> &x){
		return -log_posterior(x[0], x[1]);
	}

public:
	double alpha_map;
	double beta_map;

	double log_posterior(double alpha, double beta){
		if(alpha <= 0.0 || beta <= 0.0) return -1.0e30; //Some very small value
		int n_buttons = buttons.size();
		double tmp = log_prior(alpha, beta);
		for(int i = 0; i < n_buttons; i++)
			tmp += beta_frac_log(alpha, beta, buttons[i][1], buttons[i][0] - buttons[i][1]);
		return tmp;
	}

	Bernoulli_Hierarchy(double prior_ki = 1.0): Nelder_Mead(2), prior_k(prior_ki), prior_theta(1.0), alpha_map(1.0), beta_map(1.0) {};

	void clear(void){prior_theta = 1.0; alpha_map = beta_map = 1.0; buttons.clear();}

	void add_button(int n, int k){
		if(n < k){
			std::cout << "Wrong button inputs" << std::endl;
			throw("Wrong button inputs");
		}
		std::vector<int> btn = {n, k};
		buttons.push_back(btn);
		prior_theta = buttons.size()/(2.0*prior_k);
	}

	void add_win(void){
		if(buttons.size() == 0) add_button(0, 0);
		int indx = buttons.size()-1;
		buttons[indx][0] += 1;
		buttons[indx][1] += 1;
	}

	void add_fail(void){
		if(buttons.size() == 0) add_button(0, 0);
		int indx = buttons.size()-1;
		buttons[indx][0] += 1;
	}

	void add_button_win(void){
		std::vector<int> btn = {1, 1};
		buttons.push_back(btn);
		prior_theta = buttons.size();
	}

	void add_button_fail(void){
		std::vector<int> btn = {1, 0};
		buttons.push_back(btn);
		prior_theta = buttons.size();
	}

	double find_map(void){
		std::vector<double> init = {1.0, 1.0};
		set_x(init, 0.5);
		minimize(false);
		alpha_map = min_x[0]; beta_map = min_x[1];
		return min_val;
	}
};

#endif