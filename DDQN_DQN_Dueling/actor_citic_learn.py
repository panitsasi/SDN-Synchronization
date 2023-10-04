#networks

import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class ActorCriticNetwork(keras.Model):
    
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=512,
                name='actor_critic', chkpt_dir='tmp/actor_critic'):
        super(ActorCriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ac')
        
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activiation = 'relu')
        self.v = Dense(1, activation=None)
        self.pi = Dense(n_actions, activation = 'softmax')
        
    def call(self,state):
        value = self.fc1(state)
        value = self.fc2(value)
        
        v = self.v(value)
        pi = self.pi(value)
        return v,pi
    
#actor critic

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
#might need to get actor_critic class

class Agent:
    def __init__(self, alpha=0.003, gamma=0.99, n_actions=2):
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]
        
        self.actor_critic = ActorCriticNetwork(n_actions = n_actions)
        self.actor_critic_compile(optimizer=Adam(learning_rate=alpha))
        
    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        _, probs = self.actor_critic(state)
        
        action_probabilities = tfp.distributions.Categorial(probs=probs)
        action = action_probabilities.samples()
        self.action = action
        
        return action.numpy()[0]
    

    def learn(self, state, reward, state_, done):
        state = tf.convert_to_tensor([state],dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_],dtype=tf.float32)
        reward = tf.convert_to_tensor(reward,dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as tape:
            state_value, probs = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)
            
            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(self.action)
            
            delta = reward + self.gamma*state_value_(1-int(done)) - state_value 
            actor_loss = -log_prob*delta
            critic_loss = delta**2

            total_loss = actor_loss + critic_loss
            
            #.............more needed, gradient and learn steps
            
    

