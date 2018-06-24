import tensorflow as tf
import numpy
import gym

env=gym.make("SpaceInvaders-v0")
observation=env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    print(_)



class DeepQ():
    def __init__(self,training_data,training_labels,test_data,test_labels,game="SpaceInvaders-v0"):
        self.x=training_data
        self.y=training_labels
        self.test_x=test_data
        self.test_y=test_labels
        self.game=game
        self.env=gym.make(game)# setting up game enviroment


    def Q_Network(self,learning_rate=1e-3):
        """
        Basic Q learning structure.
        """
        pass




    def Q_Learning(self,train_x,train_y):
        """
        Train the Q Network
        Parameters:
        train_x: training data
        train_y: the trainig labels
        """
        pass

    def Q_predict(self,x,model_path="Your path here"):
        """Self explaining
        Parms:
        x: input
        modell_path: The path to the saved modell
        """
        pass
