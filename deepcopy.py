
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import nengo
import nengo_dl
from nengo.dists import Uniform
from nengo.utils.ensemble import response_curves
import numpy as np
import pickle
from nengo.solvers import LstsqL2
import MyLearning
import DecoderLearning
import RecurrentLearning
#import MyLearningDL
from clsgeneral import *
import copy


#--------------------------
#-------------------------------
def generateData(n):
    """ 
    generates a 2D dataset with n samples. 
    The third element of the sample is the label
    """
    xb = (np.random.rand(n)*2-1)/2+0.5
    yb = (np.random.rand(n)*2-1)/2+0.5
    xg = (np.random.rand(n)*2-1)/2-0.5
    yg = (np.random.rand(n)*2-1)/2-0.5
    
    inputs = []
    for i in range(len(xb)):
        inputs.append([xb[i],yb[i],[0,1]])
        inputs.append([xg[i],yg[i],[1,0]])
    return inputs


def plot(sim):
    fig = plt.figure(figsize=(18,18))
    p1 = fig.add_subplot(4,1,1)
    p1.plot(t, sim.data[answer_p])
    p1.set_title("Answer")
    p1.set_xlim(int(factor*samples)-10,sim_time)
    p1.set_ylim(-0.2,1.2)

    p2 = fig.add_subplot(4,1,2)
    p2.plot(t, sim.data[correct_answer_p])
    p2.set_title("Correct Answer")
    p2.set_xlim(int(factor*samples)-10,sim_time)

    p3 = fig.add_subplot(4,1,3)
    p3.plot(t, sim.data[error_p])
    p3.set_title("Error")
    p3.set_xlim(int(factor*samples)-10,sim_time)

    p4 = fig.add_subplot(4,1,4)
    p4.plot(t, sim.data[actual_error])
    p4.set_title("Acutal Error")
    p4.set_xlim(int(factor*samples)-10,sim_time)

    #------------------------------------
    print("hi")


def getdicIndexbykey(searchdic,searchkey,type=1):
        try:
            #searchlst=list(searchdic.keys())
            i=0
            if (type==1):
                for x in searchdic.keys():
                    try:
                        if (str(x)==str(searchkey)):
                            break
                    finally:
                        i=i+1
            return i-1
        except:
            return -1


samples = 10
data = generateData(samples)
N = 10 
D_inp = 2
D_out = 2
factor =1
sim_time = samples
model = nengo.Network('net1')
#weights = np.random.randn(D_inp,D_out).transpose()  
#weights = np.zeros((D_inp,D_out)).transpose()
weights = np.zeros((N*D_inp,N*D_out)).transpose()

with model:  
    
    def stim(t):
        for i in range(samples):
            if int(t) % samples <= i:
                return [data[i][0], data[i][1]]
        return 0
    
        
    def stim_ans(t):
        for i in range(samples):
            if int(t) % samples <= i:
                return data[i][2]
        return 0   

    stim = nengo.Node(output=stim, size_out=D_inp, label="stim")
    stim_ans = nengo.Node(output=stim_ans, size_out=D_out, label="stim_ans")

    input = nengo.Ensemble(N*D_inp, dimensions=D_inp, label="input") 
    answer = nengo.Ensemble(N*D_out, dimensions=D_out, label="answer")
    #answer = nengo.Node(size_out=D_out, label="answer")
    correct_answer = nengo.Ensemble(N*D_out, dimensions=D_out, radius=2, label="correct_answer") 

    nengo.Connection(stim, input) 
    nengo.Connection(stim_ans, correct_answer)

    error = nengo.Ensemble(N*D_out, dimensions=D_out, radius=2, label="error")  
    nengo.Connection(answer, error, transform=1)
    nengo.Connection(correct_answer, error, transform=-1)

    #nengo.Connection(answer.neurons, answer.neurons, transform=np.random.uniform(low=-1,high=1, size=(N*D_inp,N*D_inp)))
    #nengo.Connection(input.neurons, input.neurons,transform=np.random.uniform(low=-1,high=1, size=(N*D_inp,N*D_inp)))
    #nengo.Connection(input.neurons, input.neurons,transform=np.zeros(shape=(N*D_inp,N*D_inp)))

    actual_error = nengo.Ensemble(N*D_out, dimensions=D_out, label="actual_error")
    nengo.Connection(answer, actual_error, transform=1)
    nengo.Connection(correct_answer, actual_error, transform=-1)

    #rec = nengo.Ensemble(N*D_inp , dimensions=D_inp, label="answer")
    #connEnc = nengo.Connection(input.neurons, rec.neurons)
    #connEnc1 = nengo.Connection(rec.neurons,input.neurons, transform=np.random.randn(N*D_inp,N*D_inp).transpose())
    #, transform=weights
    conn = nengo.Connection(input, answer) 
    
    conn.solver = LstsqL2(weights=False)
    conn.learning_rule_type = nengo.PES()  
    #conn.learning_rule_type = DecoderLearning.MyDecoderLearning()
    #conn.learning_rule_type = RecurrentLearning.MyRecurrentLearning()
   
    error_conn = nengo.Connection(error, conn.learning_rule)


    def inhibit(t):
        return 2.0 if t > int(factor*samples) else 0.0
        #return 0.0 
     
    inhib = nengo.Node(inhibit,label="inhibit")
    nengo.Connection(inhib, error.neurons, transform=[[-1]] * error.n_neurons)

#----------------------------------
with model:
    input_p = nengo.Probe(input, synapse=0.1)
    answer_p = nengo.Probe(answer, synapse=0.1)
    correct_answer_p = nengo.Probe(correct_answer, synapse=0.1)
    error_p = nengo.Probe(error, synapse=0.1)   
    actual_error = nengo.Probe(actual_error, synapse=0.1)
    weights_p = nengo.Probe(conn, 'weights', synapse=0.01, sample_every=0.01)


sim = nengo.Simulator(model)
sim.run_steps(5*1000)
t = sim.trange()
plot(sim)

###############################################

simcopy=copy.deepcopy(sim)
simcopy.run_steps(5*1000)
t = simcopy.trange()
plot(simcopy)


###############################################

#state1=sim.__getstate__()

#sim0 = nengo.Simulator(model)
#sim0.__setstate__(copy.deepcopy(state1))
#sim0.run_steps(5*1000)
#plot(sim0)

###############################################

#state1=sim.__getstate__()
#with open('.\state1.dat', 'wb') as f:
#    pickle.dump(state1, f)

#with open('.\state1.dat', 'rb') as f:
#    state2==pickle.load(f)

#sim2 = nengo.Simulator(model)
#sim2.__setstate__(state2)
#sim2.run_steps(5*1000)
#plot(sim2)

###############################################

#with open('.\sim3.dat', 'wb') as f:
#    pickle.dump(sim, f)

#with open('.\sim3.dat', 'rb') as f:
#    sim3=pickle.load(f)

#sim3.run_steps(5*1000)


sim.run_steps(5*1000)
t = sim.trange()
plot(sim)

print("end")
