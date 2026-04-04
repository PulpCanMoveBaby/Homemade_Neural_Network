import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from datetime import datetime


#USER INPUT
input_data = [[7,3,16,11], [5,6,19,16], [8,5,24,15], [5,4,16,11], [12,4,15,9], [11,6,11,8], [14,2,18,12]]
observed_data = [[0,1,1],[1,0,1],[1,0,1],[0,0,1],[1,1,1],[0,0,0],[1,0,1]]
learning_rate = 0.0000001
num_nodes = 20


input_dim = len(input_data[0])
output = len(observed_data[0])


assert len(input_data) == len(observed_data), 'input and observed must be the same length'

weights_before_activation = [float(np.random.normal()) for _ in range(num_nodes*input_dim)]
weights_after_activation = np.random.normal(size=num_nodes)

#b1, b2, b3----------------
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
    # predicted = [predicted[i:i+output] for i in range(0,len(predicted)+output,output)]
    return [sum((i[n]-j[n])**2 for i,j in zip(observed, predicted)) for n in range(output)]

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

    #activation function-----------------
    act_funcs = [softplus(i) for i in eq_to_1]  


    #final step from hidden layer to output-----------------------------
    predicted_no_bias_added = [[act_funcs[j][i]*weights_after_activation[j]
                                for j in range(len(act_funcs))]
                                for i in range(len(act_funcs[0]))]
    

    predicted_and_summed = [sum(i) for i in predicted_no_bias_added]
    predicted_with_bias = [i+j for i in predicted_and_summed for j in biases[num_nodes:]]
    predicted_with_bias = [predicted_with_bias[i:i+output] for i in range(0,len(predicted_with_bias),output)]
    error = SSR(observed_data, predicted_with_bias)

    #take derivatives---------------------------
    dSSR_dPred = [sum([-2*(observed_data[i][j]-predicted_with_bias[i][j]) for j in range(len(observed_data[0]))]) for i in range(len(observed_data))]

    #first layer-------------------------------
    w_layer1 = [[sum([dSSR_dPred[i]*weights_after_activation[k]*d_softplus(h)[i]*input_data[i][j] 
                for i in range(len(input_data[0]))])
                for j in range(len(input_data[0]))] 
                for k,h in enumerate(eq_to_1)]
    
    b_layer1 = [sum([dSSR_dPred[i]*weights_after_activation[k]*d_softplus(h)[i] 
                for i in range(len(input_data))]) for k,h in enumerate(eq_to_1)]

    #second layer --------------------------------
    w_layer2 = [sum([dSSR_dPred[i]*d_softplus(h)[i] for i in range(len(input_data))]) for h in eq_to_1]
    b_layer2 = [sum([dSSR_dPred[i]*1 for i in range(len(input_data))])for _ in range(output)]

    differentiated_variables_list = w_layer1+w_layer2+b_layer1+b_layer2

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
last_sum_error = 100000000000
new_low = 1000000000000

#get the previous low from ./nn_{input_dim}_{num_nodes}_{output}.txt' file in this directory
previous_low = 44.721588659499254

num_trials = 100000
random_check = [0,0]
for epoch in range(num_trials):
    new_vars, predicted_with_bias, summed_and_learned, error = trial(new_vars)
    
    if output == 1:
        if epoch%int(num_trials/4) == 0:
            plt.plot(observed_data, label = 'Observed')
            plt.plot(predicted_with_bias, label = 'Predicted')
            plt.legend()
            plt.show()
    elif sum(error) < previous_low:
        # # print(f'@ {epoch} trials, {sum(error)} is greater than {last_sum_error}')
        # if not choice:
        #     choice =  input('Write the previous variables to file? ')

        with open(f'./nn_{input_dim}_{num_nodes}_{output}.txt','w') as f:
            f.write(str(new_vars)+'\n')
            # f.write(str(input_data)+'\n')
            f.write(str(epoch)+'\n')
            f.write(str(sum(error))+'\n\n')
            previous_low = sum(error)
    
        if epoch== num_trials/2:
            evaluation = input("Is it getting better? ")
            if evaluation == 'n':
                sys.exit(0)
    
    elif random_check[1] + 250 == epoch and sum(error) > random_check[0]:
        print("Error is increasing, exiting now...")
        sys.exit(0)
    
    if epoch%500 == 0:
        random_check = [sum(error), epoch]
    
    print(sum(error), epoch)
    last_sum_error = sum(error)
    last_variables = new_vars

if output == 1:
    plt.plot(observed_data, label = 'Observed')
    plt.plot(predicted_with_bias, label = 'Predicted')
    choice = input('Was the fit improving? ')
    if choice.lower() == 'y':
        with open('./nn_onelayer_threeoutput_params.txt', 'a') as f:
            f.write(str(new_vars))
            f.write(f'------> error @ {num_trials} = {error}')
        now = datetime.now()
        # plt.plot(observed_data, label = 'Observed')
        # plt.plot(predicted_with_bias, label = 'Predicted')
        plt.savefig(f'./nn_onelayer_threeoutput_{now}.png')
    plt.legend()
    plt.show()
