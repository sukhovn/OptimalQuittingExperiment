#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip> 
#include <time.h>

#ifndef NELDER_MEAD_H
#define NELDER_MEAD_H

#define SWAP(a, b, c) (a) = (b); (b) = (c); (c) = (a);

template <class T>
static std::vector<T>& operator+=(std::vector<T> &a, const std::vector<T>& b){
	if(a.size() != b.size()){
		std::cout << "Vector sizes in += don't match\n";
		throw("Vector sizes in += don't match");
	}
	for(int i = 0; i < a.size(); i++)
		a[i] += b[i];
	return a;
}

template <class T>
static std::vector<T>& operator/=(std::vector<T> &a, const T mult){
	for(int i = 0; i < a.size(); i++)
		a[i] /= mult;
	return a;
}

template <class T>
static std::vector<T> operator+(const std::vector<T> &a, const std::vector<T>& b){
	if(a.size() != b.size()){
		std::cout << "Vector sizes in - don't match\n";
		throw("Vector sizes in - don't match");
	}
	std::vector<T> result(a);
	result += b;
	return result;
}

template <class T>
static std::vector<T> operator-(const std::vector<T> &a, const std::vector<T>& b){
	if(a.size() != b.size()){
		std::cout << "Vector sizes in - don't match\n";
		throw("Vector sizes in - don't match");
	}
	std::vector<T> result(a);
	for(int i = 0; i < a.size(); i++)
		result[i] -= b[i];
	return result;
}

template <class T>
static std::vector<T> operator*(const T mult, const std::vector<T> &a){
	std::vector<T> result(a);
	for(int i = 0; i < a.size(); i++)
		result[i] *= mult;
	return result;
}

template <class T>
static double vec_abs(const std::vector<T> &vec){
	double tmp = 0.0;
	for(int i = 0; i < vec.size(); i++)
		tmp += vec[i]*vec[i];
	return std::sqrt(tmp);
}

class Nelder_Mead{
public:
	const int nvar;

	virtual double function(const std::vector<double> &x) = 0;

protected:
	std::vector<std::vector<double>> x_array;
	std::vector<double> vals;

	int max_iter;
	double max_side;

private:
	double nm_alpha;
	double nm_gamma;
	double nm_rho;
	double nm_sigma;

	int indx;// index of the last element changed (unchanged in case of shrink)
	char step_type;// r - reflection, e - expansion, c - contraction, s - shrink

	void sort_points(void){//Bubble sort
		double tmp_val;
		std::vector<double> tmp_x;
		bool unsorted = true;
		while(unsorted){
			unsorted = false;
			for(int i = nvar; i > 0; i--){
				if(vals[i-1] > vals[i]){
					SWAP(tmp_val, vals[i-1], vals[i])
					SWAP(tmp_x, x_array[i-1], x_array[i])
					if(indx == i) indx = i-1;
					else if(indx == i-1) indx = i;
					unsorted = true;
				}
			}
		}
		return;
	}

public:
	const std::vector<double> &min_x;
	const double &min_val;

	//Periodically maps x between bound0 and bound1 for finite domain functions
	void box(std::vector<double> &x, const std::vector<double> &bound_st, const std::vector<double> &bound_end){
		if(x.size() != bound_st.size() || x.size() != bound_end.size()){
			std::cout << "Different domain sizes in box rountine\n";
			throw("Different domain sizes in box");
		}
		double bound0, bound1;
		for(int i = 0; i < x.size(); i++){
			bound0 = bound_st[i];
			bound1 = bound_end[i];
			while(x[i] < bound0 || x[i] > bound1){
				if(x[i] < bound0) x[i] = 2.0*bound0-x[i];
				if(x[i] > bound1) x[i] = 2.0*bound1-x[i];
			}
		}
		return;
	}

	double largest_side(void){
		double tmp, max = 0.0;
		std::vector<double> tmp_vec;

		for(int i = 0; i < nvar; i++){
			for(int j = i+1; j <= nvar; j++){
				tmp_vec = x_array[i] - x_array[j];
				if((tmp = vec_abs(tmp_vec)) > max) max = tmp;
			}
		}

		return max;
	}

	void nm_step(void){
		//Finding centroid
		std::vector<double> x_centr(nvar, 0.0);
		for(int i = 0; i < nvar; i++)
			x_centr += x_array[i];
		x_centr /= 1.0*nvar;

		std::vector<double> x_tmp = x_centr + nm_alpha * (x_centr - x_array[nvar]);
		double v_tmp = function(x_tmp);

		indx = 0;
		for(; indx < nvar; indx++) if(v_tmp < vals[indx]) break;

		// for(int i = 0; i < nvar; i++)
		// 	std::cout << x_tmp[i] << ' ';
		// std::cout << "\n" << v_tmp << std::endl;
		// std::cout << "indx is " << indx << std::endl;

		if(indx < nvar){
			//Shifting points with the higher value than v_tmp
			for(int i = nvar; i > indx; i--){
				x_array[i] = x_array[i-1];
				vals[i] = vals[i-1];
			}

			if(indx == 0){//Expansion
				// std::cout << "Expansion case\n";
				std::vector<double> x_exp = x_centr + nm_gamma * (x_tmp - x_centr);
				double v_exp = function(x_exp);
				step_type = 'e';
				if(v_exp < v_tmp){
					// std::cout << "Using expanded result\n";
					x_array[0] = x_exp;
					vals[0] = v_exp;
					return;
				}
				else{
					// std::cout << "Using reflected result\n";
					x_array[0] = x_tmp;
					vals[0] = v_tmp;
					return;
				}
			}
			else{//Reflection
				// std::cout << "Reflection case\n";
				x_array[indx] = x_tmp;
				vals[indx] = v_tmp;
				step_type = 'r';
				return;
			}
		}
		else{//Contraction
			// std::cout << "Contraction case\n";
			x_tmp = x_centr + nm_rho * (x_array[nvar] - x_centr);
			v_tmp = function(x_tmp);
			if(v_tmp < vals[nvar]){
				x_array[nvar] = x_tmp;
				vals[nvar] = v_tmp;
				sort_points();
				step_type = 'c';
				return;
			}
		}
		//Shrink
		// std::cout << "Shrink case\n";
		indx = 0;
		for(int i = 1; i <= nvar; i++){
			x_array[i] = x_array[0] + nm_sigma * (x_array[i] - x_array[0]);
			vals[i] = function(x_array[i]);
		}
		sort_points();
		step_type = 's';

		return;
	}

	Nelder_Mead(int nvari): nvar(nvari), 
							x_array(nvari+1, std::vector<double> (nvar, 0.0)),
							vals(nvari+1, 0.0),
							nm_alpha(1.0), nm_gamma(2.0), nm_rho(0.5), nm_sigma(0.5),
							step_type('i'), indx(0), max_iter(1000), max_side(1.0e-8),
							min_x(x_array[0]),
							min_val(vals[0])
	{
		for(int i = 0; i < nvar; i++)
			x_array[i+1][i] += 1.0;
	};

	void set_x(const std::vector<double> &init, double scale){
		x_array[0] = init;
		for(int i = 0; i < nvar; i++){
			x_array[i+1] = init;
			x_array[i+1][i] += scale;
		}

		for(int i = 0; i <= nvar; i++)
			vals[i] = function(x_array[i]);

		sort_points();
	}

	void random_x(const std::vector<double> &x_low, const std::vector<double> &x_high, double scale = 0.1){
		srand (time(NULL));
		for(int i = 0; i < nvar; i++){
			x_array[0][i] = x_low[i] + (x_high[i] - x_low[i])*(scale + (1.0 - 2.0*scale)*rand()/(1.0*RAND_MAX));
		}

		std::vector<double> x_range = x_high - x_low;
		for(int i = 0; i < nvar; i++){
			x_array[i+1] = x_array[0];
			x_array[i+1][i] += scale*x_range[i];
		}

		for(int i = 0; i <= nvar; i++)
			vals[i] = function(x_array[i]);

		sort_points();
	}

	int last_change_index(void) const{return indx;} //returns -1 if shrink

	void print(int num){
		std::vector<double> &x_min = x_array[num];
		for(int i = 0; i < nvar; i++)
			std::cout << x_min[i] << ' ';
		std::cout << "\n" << function(x_min) << std::endl;
	}

	void set_max_iter(int i) {max_iter = i;}
	void set_max_size(double max) {max_side = max;}

	double minimize(bool messages = true){
		int i;

		for(i = 0; i < max_iter; i++){
			nm_step();
			if(largest_side() < max_side) break;
		}

		if(i < max_iter && messages){
			std::cout << "Minimization is done\n";
			std::cout << "Number of iterations: " << i << std::endl;
			std::cout << "Minimum is reached at:\n";
			for(int i = 0; i < nvar; i++){
				std::cout << x_array[0][i];
				if(i < nvar-1) std::cout << " ";
			}
			std::cout << std::endl;
			std::cout << "Minimal value reached is: " << vals[0] << std::endl;
		}
		else if(messages){
			std::cout << "Maximal number of iterations reached\n";
			std::cout << "Number of iterations: " << i << std::endl;
			std::cout << "Best minimum reached at:\n";
			for(int i = 0; i < nvar; i++){
				std::cout << x_array[0][i];
				if(i < nvar-1) std::cout << " ";
			}
			std::cout << std::endl;
			std::cout << "Best minimal value reached: " << vals[0] << std::endl;
		}

		return vals[0];
	}

	void save_simplex(const char *file_name) const{
		std::ofstream out_x;
		out_x.open(file_name);
		out_x << "vertex,";
		for(int j = 0; j < nvar; j++){
			out_x << "coord_" << j << ",";
		}
		out_x << "func_val\n";
		for(int i = 0; i <= nvar; i++){
			out_x << i << ",";
			for(int j = 0; j < nvar; j++)
				out_x << x_array[i][j] << ",";
			out_x << vals[i] << std::endl;
		}
		return;
	}

	void load_simplex(const char *file_name){
		std::ifstream file(file_name);
		if(!file){
			std::cout << "No input file in load_simplex\n";
			throw("load_simpex_fail");
		}
		std::string line;
		if(!std::getline(file, line)){
			std::cout << "Wrong input file format\n";
			throw("load_simpex_fail");
		}
		std::istringstream iss(line);
		if(iss.peek() == EOF){
			std::cout << "Wrong input file format\n";
			throw("load_simpex_fail");
		}
		std::string token;
		std::getline(iss, token, ',');
		if(token != "vertex"){
			std::cout << "Wrong input file format\n";
			throw("load_simpex_fail");
		}
		int ndim = 0;
		while(iss.peek() != EOF){
			std::getline(iss, token, ',');
			if(token.find("coord_") != std::string::npos) ndim++;
			else if(token == "func_val") break;
			else{
				std::cout << "Wrong input file format\n";
				throw("load_simpex_fail");
			}
		}
		if(ndim != nvar){
			std::cout << "Wrong number of dimensions in input file\n";
			throw("load_simpex_fail");
		}
		std::string::size_type sztp;
		for(int i = 0; i <= nvar; i++){
			if(!std::getline(file, line)){
				std::cout << "Not enough simplex vertices\n";
				throw("load_simpex_fail");
			}
			std::istringstream iss(line);
			std::getline(iss, token, ',');
			for(int j = 0; j < nvar; j++){
				std::getline(iss, token, ',');
				if(token.empty()){
					std::cout << "Not enough simplex coordinates\n";
					throw("load_simpex_fail");
				}
				x_array[i][j] = std::stof(token, &sztp);
			}
			std::getline(iss, token, ',');
			if(token.empty()){
				std::cout << "No function value given\n";
				throw("load_simpex_fail");
			}
			vals[i] = std::stof(token, &sztp);
		}

		sort_points();
	}

	double minimize_track_csv(const char *file_name, bool messages = true){
		int i;
		std::ofstream out_x;
		out_x.open(file_name);
		out_x << "iter,step_type,func_val,";
		for(int j = 0; j < nvar; j++){
			out_x << "coord_" << j;
			if(j < nvar-1) out_x << ",";
			else out_x << std::endl;
		}

		for(i = 0; i <= nvar; i++){
			out_x << "0,i," << vals[i] << ",";
			for(int j = 0; j < nvar; j++){
				out_x << x_array[i][j];
				if(j < nvar-1) out_x << ",";
				else out_x << std::endl;
			}
		}

		for(i = 0; i < max_iter; i++){
			nm_step();

			if(step_type != 's'){
				out_x << i << "," << step_type << "," << vals[indx] << ",";
				for(int j = 0; j < nvar; j++){
					out_x << x_array[indx][j];
					if(j < nvar-1) out_x << ",";
					else out_x << std::endl;
				}
			}
			else{
				for(int k = 0; k <= nvar; k++){
					if(k == indx) continue;
					out_x << i << "," << step_type << "," << vals[k] << ",";
					for(int j = 0; j < nvar; j++){
						out_x << x_array[k][j];
						if(j < nvar-1) out_x << ",";
						else out_x << std::endl;
					}
				}
			}

			if(largest_side() < max_side) break;
		}

		if(i < max_iter && messages){
			std::cout << "Minimization is done\n";
			std::cout << "Number of iterations: " << i << std::endl;
			std::cout << "Minimum is reached at:\n";
			for(int i = 0; i < nvar; i++){
				std::cout << x_array[0][i];
				if(i < nvar-1) std::cout << " ";
			}
			std::cout << std::endl;
			std::cout << "Minimal value reached is: " << vals[0] << std::endl;
		}
		else if(messages){
			std::cout << "Maximal number of iterations reached\n";
			std::cout << "Number of iterations: " << i << std::endl;
			std::cout << "Best minimum reached at:\n";
			for(int i = 0; i < nvar; i++){
				std::cout << x_array[0][i];
				if(i < nvar-1) std::cout << " ";
			}
			std::cout << std::endl;
			std::cout << "Best minimal value reached: " << vals[0] << std::endl;
		}

		out_x.close();

		return vals[0];
	}
};

#endif