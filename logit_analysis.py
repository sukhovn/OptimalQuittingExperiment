import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.optimize import fmin
import emcee
import corner


def convert_transition_table(transition_table, **kwargs):
    cutoff = 0
    if 'max_press' in kwargs:
        cutoff = kwargs.pop('max_press')
    if(len(kwargs) > 0):
        raise TypeError(list(kwargs.keys())[0] + ' is an invalid keyword argument')
        
    wins = transition_table.shape[0]
    fails = transition_table.shape[1]
    table = np.c_[np.repeat(np.arange(wins),fails),np.tile(np.arange(fails),wins),np.reshape(transition_table,(-1))]
    table = table[table[:,2] != 0]
    if(cutoff > 0):
        table = table[table[:,2] < cutoff]
    return table


def select_n_decisions(decisions, n_stay, n_leave, **kwargs):
    return_cross = False
    if 'return_cross' in kwargs:
        return_cross = kwargs.pop('return_cross')
    if(len(kwargs) > 0):
        raise TypeError(list(kwargs.keys())[0] + ' is an invalid keyword argument')
        
    rng = np.random.default_rng()
    stay_decisions = decisions[decisions[:,2] == 0]
    leave_decisions = decisions[decisions[:,2] == 1]
    stay_index_select = rng.choice(len(stay_decisions), size=n_stay, replace=False)
    leave_index_select = rng.choice(len(leave_decisions), size=n_leave, replace=False)
    stay_select = stay_decisions[stay_index_select][:,[0,1,3]]
    leave_select = leave_decisions[leave_index_select][:,[0,1,3]]
    
    if return_cross:
        stay_not_select = np.full(len(stay_decisions), True)
        stay_not_select[stay_index_select] = False
        stay_not_select = stay_decisions[stay_not_select][:,[0,1,3]]
        leave_not_select = np.full(len(leave_decisions), True)
        leave_not_select[leave_index_select] = False
        leave_not_select = leave_decisions[leave_not_select][:,[0,1,3]]
        return (stay_select, leave_select), (stay_not_select, leave_not_select)
    else:
        return (stay_select, leave_select)
    

def plot_regression(decisions, model, **kwargs):
    if 'max_wins' in kwargs:
        max_wins = kwargs.pop('max_wins')
    else:
        max_wins = 10
    if 'max_fails' in kwargs:
        max_fails = kwargs.pop('max_fails')
    else:
        max_fails = 10
    if 'max_presses' in kwargs:
        max_presses = kwargs.pop('max_presses')
    else:
        max_presses = 100
    if 'n_plot' in kwargs:
        n_plot = kwargs.pop('n_plot')
    else:
        n_plot = 50
    if 'transition_table' in kwargs:
        transition_table = kwargs.pop('transition_table')
        transition_table = transition_table[:max_wins+1,:max_fails+1]
    else:
        transition_table = []
    if 'no_presses_model' in kwargs:
        no_presses_model = kwargs.pop('no_presses_model')
    if 'plot' in kwargs:
        ax = kwargs.pop('plot')
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    if(len(kwargs) > 0):
        raise TypeError(list(kwargs.keys())[0] + ' is an invalid keyword argument')
        

    #Plotting model
    Y = np.arange(0, max_fails, 0.25)
    Z = np.arange(0, max_presses, 0.25)
    Y, Z = np.meshgrid(Y, Z)
    X = - (model.map_estimate[2]*Y + model.map_estimate[3]*Z + model.map_estimate[0])/model.map_estimate[1]
    ax.plot_surface(X, Y, Z, alpha=0.3)
    if 'no_presses_model' in locals():
        X = - (no_presses_model.map_estimate[2]*Y + no_presses_model.map_estimate[0])/no_presses_model.map_estimate[1]
        ax.plot_surface(X, Y, Z, alpha=0.3)

    
    #Plottins sample decisions
    if(n_plot > 0):
        rng = np.random.default_rng()
        plot_stay = decisions[0]
        plot_leave = decisions[1]
        plot_stay = plot_stay[np.logical_and(plot_stay[:,0] <= max_wins, plot_stay[:,1] <= max_fails)]
        plot_leave = plot_leave[np.logical_and(plot_leave[:,0] <= max_wins, plot_leave[:,1] <= max_fails)]
        plot_stay = plot_stay[rng.choice(len(plot_stay), size=n_plot, replace=False)]
        plot_leave = plot_leave[rng.choice(len(plot_leave), size=n_plot, replace=False)]

        ax.scatter(plot_stay[:,0], plot_stay[:,1], plot_stay[:,2], c = 'b')
        ax.scatter(plot_leave[:,0], plot_leave[:,1], plot_leave[:,2], c = 'r')
        
    #Plotting exact thresholds
    if(len(transition_table) > 0):
        trans_table_list = convert_transition_table(transition_table, max_press=max_presses)
        ax.scatter(trans_table_list[:,0], trans_table_list[:,1], trans_table_list[:,2])
        
    ax.set_xlim(0, max_wins)
    ax.set_ylim(0, max_fails)
    ax.set_zlim(0, max_presses)
    ax.set_xlabel('Successes')
    ax.set_ylabel('Fails')
    ax.set_zlabel('Remaining presses')
    
    return ax


class BinaryLogisticRegression:
    def __init__(self, first_class, second_class, **kwargs):
        if 'prior_sigma' in kwargs:
            self.prior_sigma = np.array(kwargs.pop('prior_sigma'))
        else:
            self.prior_sigma = 1.0
        if(len(kwargs) > 0):
            raise TypeError(list(kwargs.keys())[0] + ' is an invalid keyword argument')
        self.first_class = first_class
        self.second_class = second_class
        self.ndim = self.first_class.shape[1]+1
        
    def log_likelihood(self, x):
        return - np.sum(np.log(1.0 + np.exp(x[0] + self.first_class @ x[1:]))) - \
                np.sum(np.log(1.0 + np.exp(-(x[0] + self.second_class @ x[1:]))))

    def log_prior(self, x):
        return - 0.5*(x @ x)/self.prior_sigma - 0.5*len(x)*np.log(2.0*np.pi*self.prior_sigma**2)
    
    def prior_transform(self, x):
        """Transforms the uniform random variables `u ~ Unif[0., 1.)`
        to the parameters of interest."""

        return scipy.stats.norm.ppf(x, scale=self.prior_sigma)
    
    def map_probability(self, class_data):
        return 1.0/(1.0 + np.exp(-(self.map_estimate[0] + class_data @ self.map_estimate[1:])))
    
    def cross_validation(self, first_class, second_class, threshold = 0.5):
        probabilities = self.map_probability(first_class)
        counts_misidentified = np.count_nonzero(probabilities > threshold)
        first_class_miss_ratio = counts_misidentified/len(probabilities)
        
        probabilities = self.map_probability(second_class)
        counts_misidentified = np.count_nonzero(probabilities < threshold)
        second_class_miss_ratio = counts_misidentified/len(probabilities)
    
        print("Percentage of misidentified stay decisions: " + '%.2f' % (100*first_class_miss_ratio))
        print("Percentage of misidentified leave decisions: " + '%.2f' % (100*second_class_miss_ratio))
        
        return first_class_miss_ratio, second_class_miss_ratio
    
    def construct_roc_curve(self, first_class, second_class, **kwargs):
        if 'npoints' in kwargs:
            npoints = kwargs.pop('npoints')
        else:
            npoints = 101
        if 'plot_curve' in kwargs:
            plot_curve = kwargs.pop('plot_curve')
        else:
            plot_curve = False
        if 'plot' in kwargs:
            plot = kwargs.pop('plot')
        else:
            plot = plt
            
        first_class_prob = self.map_probability(first_class)
        second_class_prob  = self.map_probability(second_class)
        
        probability_range = np.linspace(0, 1, npoints)
        roc_curve = []
        for pr in probability_range:
            roc_curve.append([np.count_nonzero(first_class_prob < pr)/len(first_class_prob),
                              np.count_nonzero(second_class_prob > pr)/len(second_class_prob)])

        roc_curve = np.array(roc_curve)
        if(plot_curve):
            plot.plot(roc_curve[:,0], roc_curve[:,1], **kwargs)
        return roc_curve
    
    def fit(self, **kwargs):
        if 'initial_set' in kwargs:
            initial_set = np.array(kwargs.pop('initial_set'))
        else:
            initial_set = np.zeros(self.first_class.shape[1]+1)
        if 'ftol' in kwargs:
            ftol = kwargs.pop('ftol')
        else:
            ftol = 0.0001
        if(len(kwargs) > 0):
            raise TypeError(list(kwargs.keys())[0] + ' is an invalid keyword argument')

        min_data = fmin(lambda x: -self.log_likelihood(x), initial_set, ftol=ftol, disp=False, full_output=True)
        self.mle_estimate = min_data[0]
        self.max_likelihood = -min_data[1]

        print("MLE estimates:")
        print("Plane normal vector: " + str(self.mle_estimate[1:]))
        print("Plane intercept: " + str(self.mle_estimate[0]))
        print("MLE value: " + str(self.max_likelihood))
        
        min_data = fmin(lambda x: -self.log_likelihood(x)-self.log_prior(x), 
                                 initial_set, ftol=ftol, disp=False, full_output=True)
        self.map_estimate = min_data[0]
        self.max_posterior = -min_data[1]
        
        print("\nMAP estimates:")
        print("Plane normal vector: " + str(self.map_estimate[1:]))
        print("Plane intercept: " + str(self.map_estimate[0]))
        print("MAP value: " + str(self.max_posterior))
        
    def run_mcmc(self, **kwargs):
        if 'samples' in kwargs:
            samples = kwargs.pop('samples')
        else:
            samples = 5000
        if(len(kwargs) > 0):
            raise TypeError(list(kwargs.keys())[0] + ' is an invalid keyword argument')
            
        # create a small ball around the MLE the initialize each walker 
        nwalkers = 30
        pos = self.map_estimate +  1e-4 * np.random.randn(nwalkers, self.ndim)

        # run emcee
        self.mcmc_sampler = emcee.EnsembleSampler(nwalkers, self.ndim, lambda x: self.log_likelihood(x)+self.log_prior(x))
        self.mcmc_sampler.run_mcmc(pos, samples, progress=True)
        
    def mcmc_diagnostic(self):
        tau = self.mcmc_sampler.get_autocorr_time()
        print("Autocorrelation times: " + str(tau))
        
        fig, axes = plt.subplots(self.map_estimate.size, sharex=True)
        samples = self.mcmc_sampler.get_chain()
        for i in range(self.map_estimate.size):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3, rasterized=True)
            ax.set_xlim(0, 1000)

        axes[-1].set_xlabel("step number")
        plt.plot()
        plt.show()
        
    def analyze_mcmc(self, **kwargs):
        if 'discard' in kwargs:
            discard = kwargs.pop('discard')
        else:
            discard = 100
        if 'thin' in kwargs:
            thin = kwargs.pop('thin')
        else:
            thin = 50
        if 'make_corner' in kwargs:
            make_corner = kwargs.pop('make_corner')
        else:
            make_corner = True
            
        self.mcmc_samples = self.mcmc_sampler.get_chain(discard=discard, thin=thin, flat=True)

        self.mcmc_means = np.mean(self.mcmc_samples, axis=0)
        self.mcmc_std = np.std(self.mcmc_samples, axis=0)
        print("MCMC estimates:")
        print("Plane normal vector: " + str(self.mcmc_means[1:]))
        print("Plane vector deviations: " + str(self.mcmc_std[1:]))
        print("Plane intercept: " + str(self.mcmc_means[0]))
        print("Plane intercept deviation: " + str(self.mcmc_std[0]))
        labels = ["Intercept"] + ["Slope " + str(i) for i in range(self.ndim)]
        if make_corner:
            corner.corner(self.mcmc_samples, labels=labels, **kwargs)


def plot_roc_frame(ax):
    probability_range = np.linspace(0.0, 1.0, 11)
    ax.plot(probability_range, 1.0 - probability_range, 'b--')
    ax.invert_xaxis()
    ax.set_title("Reciever operating characteristic (ROC) curve")
    ax.set_xlabel("Correctly identified stay decisions proportion")
    ax.set_ylabel("Correctly identified quit decisions proportion")

    
def log_chi_cdf(x, df):
    return -0.5*x + (0.5*df - 1)*np.log(0.5*x) - np.log(scipy.special.gamma(0.5*df))


def likelihood_ratio_test(model1, model2):
    if model1.ndim > model2.ndim:
        likelihood_ratio = 2.0*(model1.max_likelihood - model2.max_likelihood)
        p_value = 1.0 - scipy.stats.chi2.cdf(likelihood_ratio, df=model1.ndim-model2.ndim)
    else:
        likelihood_ratio = 2.0*(model2.max_likelihood - model1.max_likelihood)
        p_value = 1.0 - scipy.stats.chi2.cdf(likelihood_ratio, df=model2.ndim-model1.ndim)
    print("Chi2 value is: %lf" % likelihood_ratio)
    print("P-value of likelihood ratio test is: %lf" % p_value)
    if(p_value < 1e-10):
        print("Decimal logarithm of p-value of likelihood ratio test is: %lf" % (log_chi_cdf(likelihood_ratio, np.abs(model2.ndim-model1.ndim))/np.log(10)))


def evidence_test(model1, model2):
    if model1.ndim > model2.ndim:
        evidence_difference = model1.evidence - model2.evidence
    else:
        evidence_difference = model2.evidence - model1.evidence
    print("Log evidence difference in favor of the larger model is: %lf" % evidence_difference)
    print("Evidence ratio in favor of the larger model is: %lf" % np.exp(evidence_difference))