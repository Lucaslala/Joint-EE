from keras import backend as K
from keras.engine.topology import Layer
from keras import activations
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras.layers import *
import tensorflow as tf

class GateLayer(Layer):

    def __init__(self, gate_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.gate_dim = gate_dim * 4
        super(GateLayer, self).__init__()

    def build(self, input_shape):
        assert input_shape[0][1]==input_shape[1][1]
        # Xavier 均匀分布初始化器
        self.w = self.add_weight((self.gate_dim,int(self.gate_dim/2)),
                                 initializer='glorot_uniform',
                                 name='{}_W'.format(self.name),
                                 trainable=True)
        
        self.b = self.add_weight((self.gate_dim/2,),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 trainable=True)
        super(GateLayer, self).build(input_shape)
        
    def Mask(self, inputs, input_mask=None):

        return None
                
    def call(self, x, mask=None):
        if(len(x)==2):
            sen_att_outputs, doc_att_outputs = x
        batch_size = tf.shape(sen_att_outputs)[0]
        num_steps = tf.shape(sen_att_outputs)[1]
        #print(sen_att_outputs,batch_size,num_steps)
        input = K.concatenate([sen_att_outputs,doc_att_outputs],-1)
        input = K.reshape(input, (-1,self.gate_dim))
        gate = K.bias_add(K.dot(input,self.w),self.b)
        gate = K.sigmoid(K.abs(gate))
        gate_ = K.ones_like(gate)-K.sigmoid(K.abs(gate))
        sen_att_outputs = K.reshape(sen_att_outputs,(-1,int(self.gate_dim/2)))
        doc_att_outputs = K.reshape(doc_att_outputs,(-1,int(self.gate_dim/2)))

       	output = Add()([Multiply()([sen_att_outputs,gate]), Multiply()([doc_att_outputs,gate_])])
        output = K.reshape(output,(batch_size,num_steps,int(self.gate_dim/2)))

        return output


    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], int(self.gate_dim/2))