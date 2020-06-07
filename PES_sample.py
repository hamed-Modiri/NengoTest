

import numpy as np

import nengo
import nengo_dl
from nengo.processes import WhiteSignal
from nengo.solvers import LstsqL2
import matplotlib
import matplotlib.pyplot as plt
import MyLearning
#import MyLearningDL


matplotlib.use('TkAgg')

dim=2
model = nengo.Network()
with model:
    #inp = nengo.Node(WhiteSignal(60, high=5), size_out=dim)
    inp = nengo.Node([.3,.8], size_out=dim,label="inp")
    pre = nengo.Ensemble(60, dimensions=dim,label="pre")
    conninput=nengo.Connection(inp, pre)
    post = nengo.Ensemble(60, dimensions=dim,label="post")
    conn = nengo.Connection(pre, post, function=lambda x: np.random.random(dim))
    #conn = nengo.Connection(pre, post, function=lambda x: np.zeros(dim))
    inp_p = nengo.Probe(inp)
    pre_p = nengo.Probe(pre, synapse=0.01)
    post_p = nengo.Probe(post, synapse=0.01)

    error = nengo.Ensemble(60, dimensions=dim)
    error_p = nengo.Probe(error, synapse=0.03)

    # Error = actual - target = post - pre
    nengo.Connection(post, error)
    nengo.Connection(pre, error, transform=-1)

    ###################################################
    #conn.learning_rule_type = nengo.PES()
    conn.learning_rule_type =MyLearning.MyPES()

    nengo.Connection(error, conn.learning_rule)

    #nengo.Connection(pre, conn.learning_rule)   
    #nengo.Connection(post, conn.learning_rule)
    postvoltage = nengo.Probe(post.neurons, "voltage")
    ###################################################

    weights_p = nengo.Probe(conn, 'weights', synapse=0.01, sample_every=0.01)
    weights_input = nengo.Probe(conninput, 'weights', synapse=0.01, sample_every=0.01)

    bln=False
    conn.solver = LstsqL2(weights=True)
    bln=True

with nengo.Simulator(model) as sim:
    sim.run(4.0)

plt.figure(figsize=(12, 12))
plt.subplot(3, 1, 1)
plt.plot(sim.trange(), sim.data[inp_p].T[0], c='k', label='Input')
plt.plot(sim.trange(), sim.data[pre_p].T[0], c='b', label='Pre')
plt.plot(sim.trange(), sim.data[post_p].T[0], c='r', label='Post')
plt.ylabel("Dimension 1")
plt.legend(loc='best')
plt.subplot(3, 1, 2)
plt.plot(sim.trange(), sim.data[inp_p].T[1], c='k', label='Input')
plt.plot(sim.trange(), sim.data[pre_p].T[1], c='b', label='Pre')
plt.plot(sim.trange(), sim.data[post_p].T[1], c='r', label='Post')
plt.ylabel("Dimension 2")
plt.legend(loc='best')
plt.subplot(3, 1, 3)

if bln:
    plt.plot(sim.trange(0.01), sim.data[weights_p][..., 10])
    plt.ylabel("Connection weight");
else:
    plt.plot(sim.trange(0.01), sim.data[weights_p][..., 50])
    plt.ylabel("Decoding weight")
    plt.legend(("Decoder 10[0]", "Decoder 10[1]"), loc='best');
print("rt")