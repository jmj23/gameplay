import keras
import numpy as np
import tensorflow as tf
from keras.optimizers import SGD

from directKeys import A, D, DetectClick, PressKey, S, W, click
from Qmodels import BlockModel
from PIL import Image, ImageGrab, ImageOps
import pytesseract


def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true,y_pred)

class Play2048:
    def __init__(self, learning_rate=0.1, discount=0.95, exploration_rate=1.0, iterations=10000):
        self.learning_rate = learning_rate
        self.discount = discount # How much we appreciate future reward over current
        self.exploration_rate = 1.0 # Initial exploration rate
        self.exploration_delta = 1.0 / iterations # Shift from exploration to explotation
        # possible moves
        self.moves = [W,A,S,D]
        
        # some window coordinates
        self.replay_coords = [1920, 1325]
        self.select_coords = [2045,37]
        self.board_coords = [1471, 400, 2707, 1640]
        self.score_coords = [1824, 202, 1995, 245]

        # Output is 4 channels for W,A,S,D
        self.output_chan = 4

        # Input dimensions
        self.input_shape = (256,256,1)

        # model weights
        self.weights_path = 'Play2048_weights.h5'

        # build model
        self.build_model()

    def replay_click(self):
        click(self.replay_coords[0],self.replay_coords[1])

    def window_click(self):
        click(self.select_coords[0],self.select_coords[1])
    
    def get_board(self):
        coords = self.board_coords
        img = ImageGrab.grab(bbox=(coords[0], 
                                coords[1],
                                coords[2],
                                coords[3]))
        return img
    def get_score(self):
        coords = self.score_coords
        img = ImageGrab.grab(bbox=(coords[0], 
                                coords[1],
                                coords[2],
                                coords[3]))
        img_inv = ImageOps.invert(img)
        # check for game over
        array = np.array(img_inv)
        if array.mean() < 60:
            return None        
        config_str = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789'
        score = pytesseract.image_to_string(img_inv,
                                            config=config_str)
        try:
            score = np.int(score)
        except Exception as e:
            print(e)
            print('Score received was: ',score)
            return None
        return score

    # Define tensorflow model graph
    def define_model(self):
        model = BlockModel(self.input_shape,self.output_chan)
        self.model = model.compile(SGD(lr=self.learning_rate),loss=huber_loss)
        self.model._make_prediction_function()

    # Ask model to estimate Q value for specific state (inference)
    def get_Q(self, state):
        # Model input: Single state represented by array of 5 items (state one-hot)
        # Model output: Array of Q values for single state
        inp = self.img_to_input(state)
        q_pred = self.model.predict_on_batch(inp)
        return q_pred

    def get_next_action(self, state):
        if np.random.rand() > self.exploration_rate: # Explore (gamble) or exploit (greedy)
            return self.greedy_action(state)
        else:
            return self.random_action()

    # Which action has bigger Q-value, estimated by our model (inference).
    def greedy_action(self, state):
        # argmax picks the higher Q-value and returns the index (0,1,2,3 == W,A,S,D)
        return np.argmax(self.get_Q(state))

    def random_action(self):
        choice = np.random.randint(len(self.moves))
        return choice
    
    def take_action(self,action):
        PressKey(self.moves[action])

    def img_to_input(img):
        array = np.array(img)
        inp = array[np.newaxis,...,np.newaxis].astype(np.float)
        inp /= inp.max()
        return inp

    def train(self, old_state, action, reward, new_state):
        # Ask the model for the Q values of the old state (inference)
        old_state_Q_values = self.get_Q(old_state)

        # Ask the model for the Q values of the new state (inference)
        new_state_Q_values = self.get_Q(new_state)

        # Real Q value for the action we took. This is what we will train towards.
        old_state_Q_values[action] = reward + self.discount * np.amax(new_state_Q_values)
        
        # Setup training data
        inp = self.img_to_input(old_state)
        target = old_state_Q_values[np.newaxis,...]        

        # Train
        self.model.train_on_batch(inp,target)

    def update(self, old_state, new_state, action, reward):
        # Train our model with new data
        self.train(old_state, action, reward, new_state)

        # Finally shift our exploration_rate toward zero (less gambling)
        if self.exploration_rate > 0:
            self.exploration_rate -= self.exploration_delta

if __name__ == '__main__':
    agent = Play2048(iterations=100)
    print('Created agent')
