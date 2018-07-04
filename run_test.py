import test as DQ

t=DQ.DeepQ()
t.Q_Network(checkpoint_path="C://Users//Flajt//Documents//GitHub//Deeplearning_for_starters//Atari_modells//checkpoint-6.3.ckpt")
t.itertime=50000
t.minibatch_size=5000
t.Q_Learning(modell_path="C:\\Users\\Flajt\\Documents\\GitHub\\Deeplearning_for_starters\\Atari_modells\\SpaceInvaders-6.3.tfl")
#t.Q_predict(model_path="C:\\Users\\Flajt\\Documents\\GitHub\\Deeplearning_for_starters\\Atari_modells\\SpaceInvaders-6.tfl")

# modell six score: 25.00, before tuneing
# After tuneing up to 455
# Without flattening layer, but trained on it it reaches a max of 125
# Trainig it without flattening layer, score: 50 +/- 100
#7 gen using lstm, score: nope, not done actually
#modell 6.2 got to 520 points
#New highscore with 620 with modell six without flattening
