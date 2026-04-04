import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from datetime import datetime

input_data = [[7,3,16], [5,6,19], [8,5,24], [5,4,16], [12,4,15]]
input_dim = len(input_data[0])
observed_data = [0,1,1,0,1]
output = 1
learning_rate = 0.00001
num_nodes = 20

assert len(input_data) == len(observed_data), 'input and observed must be the same length'


weights_before_activation = [float(np.random.normal()) for _ in range(num_nodes*input_dim)]
weights_after_activation = np.random.normal(size=num_nodes)

biases = [0]*(num_nodes+output)
len_biases = len(biases)

variables = []
for var_list in [weights_before_activation, weights_after_activation, biases]:
    for var in var_list:   
        variables.append(float(var))

#funcs----------------------
def d_softplus(x):
    return [math.exp(i)/(1+math.exp(i)) for i in x]

def SSR(observed, predicted):
    return sum([(i-j)**2 for i,j in zip(observed, predicted)])

def softplus(x):
    return [math.log(1+math.exp(i)) for i in x]

def trial(variables):

    first_run = True
    if first_run:
        new_vars = variables
        first_run = False

    weights_before_activation = new_vars[:input_dim*num_nodes]
    weights_after_activation= new_vars[input_dim*num_nodes:-len_biases]
    biases = variables[-len_biases:]

    eq_to_1_no_bias = [[sum(data_point[i]*weights_before_activation[i+(k*input_dim)]
                            for i in range(len(input_data[0])))
                            for data_point in input_data] for k in range(num_nodes)]
    eq_to_1 = [[i+biases[idx] for i in eq] for idx,eq in enumerate(eq_to_1_no_bias)]

    act_funcs = [softplus(i) for i in eq_to_1]

    predicted_no_bias_added = [[act_funcs[j][i]*weights_after_activation[j] for i in range(len(act_funcs[0]))]
                                for j in range(len(act_funcs))]
    predicted_and_summed = [sum(eq[i] for eq in predicted_no_bias_added) for i in range(len(predicted_no_bias_added[0]))]
    predicted_with_bias = [i+j for i in predicted_and_summed for j in biases[num_nodes:]]

    error = SSR(observed_data, predicted_with_bias)

    #take derivatives---------------------------

    dSSR_dPred = [-2*(observed_data[i]-predicted_with_bias[i]) for i in range(len(observed_data))]

    w_layer1 = [[sum([dSSR_dPred[i]*weights_after_activation[k]*d_softplus(h)[i]*input_data[i][j] 
                for i in range(len(input_data))])
                for j in range(len(input_data[0]))] for k,h in enumerate(eq_to_1)]
    b_layer1 = [sum([dSSR_dPred[i]*weights_after_activation[k]*d_softplus(h)[i] 
                for i in range(len(input_data))]) for k,h in enumerate(eq_to_1)]

    w_layer2 = [sum([dSSR_dPred[i]*d_softplus(h)[i] for i in range(len(input_data))]) for h in eq_to_1]
    b_layer2 = sum([dSSR_dPred[i]*1 for i in range(len(input_data))])

    differentiated_variables_list = w_layer1+w_layer2+b_layer1+[b_layer2]

    summed_and_learned = []
    for i in differentiated_variables_list:
        if isinstance(i,list):
            for j in i:
                summed_and_learned.append(j*learning_rate) 
        else:
            summed_and_learned.append(i*learning_rate)

    new_vars = [i-j for i,j in zip(variables, summed_and_learned)]

    return new_vars, predicted_with_bias, summed_and_learned, error

#run first trial to get the new vars variable---------------------------
new_vars, predicted_with_bias, summed_and_learned, error = trial(variables)

#run a bunch more-----------------------------------
num_trials = 20000
for epoch in range(num_trials):
    new_vars, predicted_with_bias, summed_and_learned, error = trial(new_vars)

    if epoch%int(num_trials/4) == 0:
        plt.plot(observed_data, label = 'Observed')
        plt.plot(predicted_with_bias, label = 'Predicted')
        plt.legend()
        plt.show()

    if epoch== num_trials/2:
        evaluation = input("Is it getting better? (n) to exit, any key to continue: ")
        if evaluation == 'n':
            sys.exit(0)
            
    print(error, epoch)

plt.plot(observed_data, label = 'Observed')
plt.plot(predicted_with_bias, label = 'Predicted')
choice = input('Was the fit improving? Enter (y) to write the parameters to a text file, (n) to skip.')
if choice.lower() == 'y':
    with open('./nn_onelayer_oneoutput.txt', 'a') as f:
        f.write(str(new_vars))
        f.write(f'\n\n------> error @ {num_trials} = {error}\n\n')
    now = datetime.now()
    plt.savefig(f'./nn_onelayer_oneoutput_{now}.png')
plt.legend()
plt.show()
