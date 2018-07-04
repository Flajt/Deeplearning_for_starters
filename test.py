import tflearn as tfl
import numpy as np
import gym
from collections import deque
import random
import time #to see how much time is needed
"""
Partly taken from(and changed): https://github.com/llSourcell/deep_q_learning/blob/master/03_PlayingAgent.ipynb
Other sources are DQN papers and videos, but the code isn`t taken from them only how it works!
"""
#env=gym.make("SpaceInvaders-v0")
#observation=env.reset()
#for _ in range(1000):
#    env.render()
#    action = env.action_space.sample() # your agent here (this takes random actions)
#    observation, reward, done, info = env.step(action)
#    if done==True:
#        env.reset()

###TO USE LSTM, DONT USE BATCHES###

class DeepQ():
    def __init__(self,game="SpaceInvaders-v0"):
        self.game=game
        self.env=gym.make(game)# setting up game enviroment
        self.storage=deque()
        self.filter_size=[4,4]
        self.itertime=1000
        self.random_move_prop=0.8
        np.random.seed(1)# "seed" the generator to later recreate the same outputs
        self.minibatch_size=250
        self.discounted_future_reward=0.9

    def Q_Network(self,learning_rate=0.000001,load=False,model_path=None,checkpoint_path="C://Users//Flajt//Documents//GitHub//Deeplearning_for_starters//Atari_modells//checkpoint.ckpt"):
        """
        Basic Q learning structure.
        Parameters:
            learning_rate=0.000001 (default): the learning rate for the modell
            load=False
        """
        if load==False:
            net=tfl.layers.core.input_data(shape=[None,210,160,3])# rework this stuff
            net=tfl.layers.conv.conv_2d(net,nb_filter=4,filter_size=self.filter_size,activation='relu')
            net=tfl.layers.conv.conv_2d(net,nb_filter=4,filter_size=self.filter_size,activation="relu")
            #net=tfl.layers.conv.conv_2d(net,nb_filter=8,filter_size=self.filter_size)
            #net=tfl.layers.fully_connected(net,20,activation="relu")
            #net=tfl.layers.flatten(net)
            #net=tfl.layers.fully_connected(net,18,activation="relu")
            net=tfl.layers.fully_connected(net,10,activation='relu')
            net=tfl.layers.fully_connected(net,self.env.action_space.n,activation="linear")
            net=tfl.layers.estimator.regression(net,learning_rate=learning_rate,loss="mean_square")
            self.modell=tfl.DNN(net,checkpoint_path=checkpoint_path)

        else:
            net=tfl.layers.core.input_data(shape=[None,210,160,3])# rework this stuff
            net=tfl.layers.conv.conv_2d(net,nb_filter=3,filter_size=self.filter_size,activation='relu')
            net=tfl.layers.conv.conv_2d(net,nb_filter=3,filter_size=self.filter_size,activation="relu")
            #net=tfl.layers.conv.conv_2d(net,nb_filter=8,filter_size=self.filter_size)
            #net=tfl.layers.fully_connected(net,20,activation="relu")
            #net=tfl.layers.flatten(net)
            #net=tfl.layers.fully_connected(net,18,activation="relu")
            net=tfl.layers.fully_connected(net,10,activation='relu')
            net=tfl.layers.fully_connected(net,self.env.action_space.n,activation="linear")
            net=tfl.layers.estimator.regression(net,learning_rate=learning_rate,loss="mean_square")
            self.modell=tfl.DNN(net)
            self.modell.load(model_path,weights_only=True)

    def Q_Learning(self,modell_path="C:\\Users\\Flajt\\Documents\\GitHub\\Deeplearning_for_starters\\Atari_modells\\SpaceInvaders1.tfl"):
        """
        Train the Q Network
        Parameters:
            modell_path: The path were your modell should be stored
        """
        observation=self.env.reset()# reset enviroment
        for i in range(self.itertime):
            #self.env.render() # uncomment to observ your network
            observation=observation.reshape(1,210,160,3)# reshape it, so we can fit it inside our network
            if np.random.rand()<=self.random_move_prop:# let's figure out randomly if the modell should predict or not
                #print("Random step") #for debugging usefull
                action=np.random.randint(low=0,high=self.env.action_space.n)# take an action
            else:
                #print("Random prediction") #for debugging usefull
                action=self.modell.predict(observation)# let the modell make a prediction
                action=np.argmax(action)# take the highest value the network predicts, each value repersent one move (left, right, fire a.s.o)
            new_observation, reward, done, info=self.env.step(action)# take the action that was choosen, by either the network or the random guess (it's both random guessing, because the network isn't trained yet)
            self.storage.append((observation,action,reward,new_observation,done))# save all what we have observed and save it inside here to learn later from it
            observation=new_observation# change name so the stuff works on
            if done:# if we have lost a game the enviroment will be resetted
                self.env.reset()
        print("###############################################")
        print("Done with observing!")
        print("###############################################")
        t=time.time()
        minibatch=random.sample(self.storage,self.minibatch_size)# take random observations from our data
        self.storage=None#should free some memory, dont know if it work
        x=np.zeros((self.minibatch_size,)+observation.shape)# create a input matrix
        y=np.zeros((self.minibatch_size,self.env.action_space.n))# create an output matrix (these will store our )
        for i in range(0,self.minibatch_size):# iterate through the baches
            print("Processing batch data... (step:"+str(i)+" from "+str(self.minibatch_size)+")")
            Observation=minibatch[i][0]# one is the obeservation
            Action=minibatch[i][1]# is the action that has been taken
            Reward=minibatch[i][2]# the reward we got for this action
            New_observation=minibatch[i][3]#the new observation
            done=minibatch[i][4]# if we are done or not
            x[i:i+1]=Observation.reshape((1,)+observation.shape)# we create a batch with the observation
            y[i]=self.modell.predict(Observation)#the i the position in the label matrix is the prediction of the network for this obeservation
            Q_sa=self.modell.predict(Observation)#predict without appending it
            if done:# if we won the game
                y[i,action]=reward#add for the action we have done a reward
            else:# if not
                y[i,action]=reward+self.discounted_future_reward*np.max(Q_sa)#we predict the futur reward and append it for the action
            self.modell.fit_batch(x,y)# fit the batch
        self.modell.save(modell_path)#save modell
        t_2=time.time()# time after we are done
        print("")
        print("Modell fitting acomplished!")
        print("Time needed:"+str((t_2-t)/60)+" min.")# calculate the time we needed for training
        print("")

    def Q_predict(self,model_path="Your path here"):
        """
        Parms:
        modell_path: The path to the saved modell.
        """
        self.Q_Network(load=True,model_path=model_path)#load the modell
        observation=self.env.reset()#reset enviroment
        observation=observation.reshape((1,)+observation.shape)# prepare observation
        done=False# set done to false
        total_reward=0.0# our total reward will be stored here
        while not done:# if we are not done we try to win, using this actions
            self.env.render()#show gameplay
            Q=self.modell.predict(observation)#make prediction
            action=np.argmax(Q)# choose the one with the highest probability
            new_observation,reward,done,info=self.env.step(action)# get feedback from enviroment
            observation=new_observation
            observation=new_observation.reshape((1,)+observation.shape)#reshape the observation feed it into the network
            total_reward+=reward# sum up the rewards we get while playing
        print("Game ends with a score of: "+str(total_reward))#print total reward, we have gained in the game, so you can evaluate the model
        print("")
