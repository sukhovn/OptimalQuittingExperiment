import numpy as np
import pandas as pd

class Decision_Process:
    def process_simulation(self, trial_record):
        #subjects_decisions contains subject decisions for each subject
        #the first column contains number of accumulated wins, the second contains accumulated fails
        #the third is boolean and contains 1 if subject decided to change the button and 0 if subject decided to stay
        #the fourth contains the number of remaining presses
        self.subjects_decisions = []
        #subjects_button_activity contains accumulated buttons wins, fails and remaining presses
        #for all subjects split by button
        self.subjects_button_activity = []
        for record in trial_record:
            #chunking the whole press history into separate buttons
            button_split = [record[record[:,0]==k] for k in np.unique(record[:,0])]
            #reforming into a decision set, the first column has number of wins, second has number of fails
            #the third column is 0 if the decision is to stay and 1 if it is to leave
            decisions = [np.c_[elt[:,1:3],
                               np.eye(len(elt),M=1,k=1-len(elt))] for elt in button_split]
            #recording down presses where a new button was chosen
            button_split_pos = np.cumsum([len(el) for el in decisions])[:-1]
            #concatenating decisions and adding number of remaining presses
            decisions = np.concatenate(decisions)
            #we exclude the last decision since it is always stay and thus noninformative
            decisions = np.c_[decisions, np.arange(len(decisions))[::-1]].astype(int)
            #appending the result to total record
            self.subjects_decisions.append(decisions[:-1])
            self.subjects_button_activity.append(np.split(decisions[:,[0,1,3]], button_split_pos))

        #collecting all decisions togeher and sorting it
        decisions = np.concatenate(self.subjects_decisions)
        self.decisions = decisions[np.lexsort((decisions[:,3], decisions[:,2], decisions[:,1], decisions[:,0]))]
    
    def process_experiment(self, data_frame, kwargs):
        drop_first_button = False
        if 'drop_first_button' in kwargs:
            drop_first_button = kwargs.pop('drop_first_button')
        if(len(kwargs) > 0):
            raise TypeError(list(kwargs.keys())[0] + ' is an invalid keyword argument')

        #subjects_decisions contains subject decisions for each subject
        #the first column contains number of accumulated wins, the second contains accumulated fails
        #the third is boolean and contains 1 if subject decided to change the button and 0 if subject decided to stay
        #the fourth contains the number of remaining presses
        self.subjects_decisions = []
        #subjects_button_activity contains accumulated buttons wins, fails and remaining presses for all the subjects per button
        self.subjects_button_activity = []

        for subject in data_frame['press_data']:
            #reading press history from subject's database entry
            record = np.array([[el['button_number'],el['outcome']] for el in subject])

            if(drop_first_button):
                record = record[record[:,0] != 0]
                if(len(record) == 0):
                    continue
            
            #chunking the whole press history into separate buttons
            button_split = [record[record[:,0]==k,1] for k in np.unique(record[:,0])]
            #reforming into a decision set, the first column has number of wins, second has number of fails
            #the third column is 0 if the decision is to stay and 1 if it is to leave
            decisions = [np.vstack([np.cumsum(elt),
                                        np.arange(1,len(elt)+1)-np.cumsum(elt),
                                        np.eye(1,M=len(elt),k=len(elt)-1)]).T for elt in button_split]
            #recording down presses where a new button was chosen
            button_split_pos = np.cumsum([len(el) for el in decisions])[:-1]
            #concatenating decisions and adding number of remaining presses
            decisions = np.concatenate(decisions)
            #we exclude the last decision since it is always stay and thus noninformative
            decisions = np.c_[decisions, np.arange(len(decisions))[::-1]]
            #appending the result to total record
            self.subjects_decisions.append(decisions[:-1])
            self.subjects_button_activity.append(np.split(decisions[:,[0,1,3]], button_split_pos))

        #collecting all decisions togeher and sorting it
        decisions = np.concatenate(self.subjects_decisions)
        self.decisions = decisions[np.lexsort((decisions[:,3], decisions[:,2], decisions[:,1], decisions[:,0]))].astype(int)
        
    def __init__(self, input_data, **kwargs):
        if type(input_data) == pd.core.frame.DataFrame:
            self.process_experiment(input_data, kwargs)
        else:
            self.process_simulation(input_data)
        
    def get_decisions(self, **kwargs):
        discard_no_fails = True
        wins_more_fails = False
        if 'discard_no_fails' in kwargs:
            discard_no_fails = kwargs.pop('discard_no_fails')
        if 'wins_more_fails' in kwargs:
            wins_more_fails = kwargs.pop('wins_more_fails')
        if(len(kwargs) > 0):
            raise TypeError(list(kwargs.keys())[0] + ' is an invalid keyword argument')
        
        if(discard_no_fails and wins_more_fails):
            return self.decisions[np.logical_and(self.decisions[:,0] > self.decisions[:,1], self.decisions[:,1] > 0)]
        elif discard_no_fails and not wins_more_fails:
            return self.decisions[self.decisions[:,1] > 0]
        elif wins_more_fails:
            return self.decisions[self.decisions[:,0] > self.decisions[:,1]]
        else:
            return self.decisions
        
    def analyze_largest_contribution(self):
        # Contribution (number of wins) for each button for each subject
        contributions = [[el[-1,0] for el in subject] for subject in self.subjects_button_activity]
        #Total number of button pressed
        num_buttons = np.array([len(subject) for subject in contributions], dtype=np.int32)
        #Index of the button with largest contribution
        max_button_index = np.array([subject.index(max(subject)) for subject in contributions], dtype=np.int32)
        #Number of runs where the final button was the button with the largest contribution
        num_last_largest_contr = np.count_nonzero(max_button_index==(num_buttons-1))
        #Ratio of largest button contribution to total contribution
        top_button_to_all = np.array([max(subject)/sum(subject) for subject in contributions])
        
        percentiles = np.percentile(num_buttons, [25, 50, 75])
        print("The average number of buttons explored is " + str(np.sum(num_buttons)/len(num_buttons)))
        print("The median number of buttons explored is %.d (IQR %.1lf-%.1lf)" % (percentiles[1], percentiles[0], percentiles[2]))
        print("The average reward received is " + 
                      str(np.sum([sum(subject) for subject in contributions])/len(contributions)))
        p_largest = num_last_largest_contr/len(contributions)
        print("Out of " + str(len(contributions)) + " subjects in total " + str(num_last_largest_contr) +
                    " subjects (or " + "%.3lf%% +- %.3lf%%" % (100*p_largest, np.sqrt(p_largest*(1-p_largest)/len(contributions))) +
                      ") got the largest contribution from the last button")
        perc_top_to_all = np.percentile(top_button_to_all, [25, 50, 75])
        print("The average contribution of the button with the largest contribution is " +
                  "%4.2f" % (100*np.mean(top_button_to_all)) + " %")
        print("The median contribution of the button with the largest contribution is %.2lf (IQR %.2lf-%.2lf)" %\
              (perc_top_to_all[1], perc_top_to_all[0], perc_top_to_all[2]))
        
        return num_buttons, top_button_to_all

def get_wins_fails(decisions):
    stay = decisions[decisions[:,2] == 0]
    leave = decisions[decisions[:,2] == 1]

    print("Total number of stay decisions is " + str(len(stay)))
    print("Total number of leave decisions is " + str(len(leave)))

    return [leave[:,3], stay[:,3]]

def aggregate_wins_fails(decisions):
    stay = decisions[decisions[:,2] == 0]
    leave = decisions[decisions[:,2] == 1]
    leave_presses, leave_counts = np.unique(leave[:,3], axis=0, return_counts=True)
    stay_presses, stay_counts = np.unique(stay[:,3], axis=0, return_counts=True)
    common_presses = np.intersect1d(leave_presses,stay_presses)

    common = []
    for el in common_presses:
        row_tmp = [el, leave_counts[leave_presses == el][0], stay_counts[stay_presses == el][0]]
        common.append(row_tmp)
    return np.array(common)

# Newer version of the previous function
def aggregate_by_remaining_presses(decisions):
    stay = decisions[decisions[:,-2] == 0]
    leave = decisions[decisions[:,-2] == 1]
    leave_presses, leave_counts = np.unique(leave[:,-1], axis=0, return_counts=True)
    stay_presses, stay_counts = np.unique(stay[:,-1], axis=0, return_counts=True)
    
    count_array = np.zeros((max(max(leave_presses), max(stay_presses))+1,2))
    count_array = np.c_[(np.arange(len(count_array)),count_array)]
    count_array[leave_presses, 1] = leave_counts
    count_array[stay_presses, 2] = stay_counts
    count_array = count_array[(count_array != 0)[:,1:].any(axis=1)]
    return count_array