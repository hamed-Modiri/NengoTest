import numpy as np
from nengo.builder import Builder, Operator, Signal
from nengo.builder.operator import Copy, Reset
from nengo.connection import LearningRule
from nengo.ensemble import Ensemble
from nengo.exceptions import BuildError
from nengo.learning_rules import *
from nengo.node import Node
from nengo.synapses import SynapseParam, Lowpass
from nengo.builder.learning_rules import (
    get_pre_ens,
    get_post_ens,
    get_post_ens,
    build_or_passthrough,
)


class MyPES(LearningRuleType):

    modifies = "decoders"
    probeable = ("error", "activities", "delta")

    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1e-4)
    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)

    def __init__(self, learning_rate=Default, pre_synapse=Default):
        super().__init__(learning_rate, size_in="post_state")
        if learning_rate is not Default and learning_rate >= 1.0:
            warnings.warn(
                "This learning rate is very high, and can result "
                "in floating point errors from too much current."
            )

        self.pre_synapse = pre_synapse

    @property
    def _argdefaults(self):
        return (
            ("learning_rate", PES.learning_rate.default),
            ("pre_synapse", PES.pre_synapse.default),
        )

class SimMyPES(Operator):


    def __init__(self, pre_filtered, error, delta, learning_rate, encoders=None, tag=None,pre=None,post=None):
        super(SimMyPES, self).__init__(tag=tag)

        self.learning_rate = learning_rate

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, error] + ([] if encoders is None else [encoders]) 
        self.myreads=[pre,post]       
        self.updates = [delta]
        self.curentdt=0

    @property
    def delta(self):
        return self.updates[0]

    @property
    def encoders(self):
        return None if len(self.reads) < 3 else self.reads[2]

    @property
    def error(self):
        return self.reads[1]

    @property
    def pre(self):
        return self.myreads[0]
    @property
    def post(self):
        return self.myreads[1]

    @property
    def pre_filtered(self):
        return self.reads[0]

    def _descstr(self):
        return "pre=%s, error=%s -> %s" % (self.pre_filtered, self.error, self.delta)

    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        error = signals[self.error]
        delta = signals[self.delta]
        n_neurons = pre_filtered.shape[0]
        alpha = -self.learning_rate * dt / n_neurons
        self.curentdt=self.curentdt+dt

        #########################################################################
        # I would like to get y and y_hat
        pre = signals[self.pre]            #Target (Y)
        post = signals[self.post]          #Actual (Y_hat)

        # I would like to get membrane potential of special Ensemble like "post"
        mempotential = signals[self.post]["voltage"]

        #########################################################################
        
        if self.encoders is None:
            def step_simpes():
                np.outer(alpha * error, pre_filtered, out=delta)
        else:
            encoders = signals[self.encoders]
            def step_simpes():
                #delta[...]=np.zeros(np.shape(delta))
                np.outer(alpha * np.dot(encoders, error), pre_filtered, out=delta)
                
        return step_simpes

@Builder.register(MyPES)
def build_mypes(model, pes, rule):


    conn = rule.connection

    # Create input error signal
    error = Signal(shape=rule.size_in, name="PES:error")
    model.add_op(Reset(error))
    model.sig[rule]["in"] = error  # error connection will attach here

    # Filter pre-synaptic activities with pre_synapse
    acts = build_or_passthrough(model, pes.pre_synapse, model.sig[conn.pre_obj]["out"])

    if conn.is_decoded:
        encoders = None
    else:
        post = get_post_ens(conn)
        encoders = model.sig[post]["encoders"][:, conn.post_slice]
    
    ##############################################
    pre = Signal(shape=rule.size_in, name="PES:pre")
    model.add_op(Reset(pre))
    model.sig[rule]["pre"] = pre  # target (Y)

    post = Signal(shape=rule.size_in, name="PES:post")
    model.add_op(Reset(post))
    model.sig[rule]["post"] = post  # actual (Y_hat)

    model.add_op(
        SimMyPES(
            acts, 
            error, 
            model.sig[rule]["delta"], 
            pes.learning_rate, 
            encoders=encoders,
            tag=None,
            pre=pre,
            post=post
        )
    )

    # expose these for probes
    model.sig[rule]["error"] = error
    model.sig[rule]["activities"] = acts
    model.sig[rule]["pre"] = pre
    model.sig[rule]["post"] = post

###################################################################
