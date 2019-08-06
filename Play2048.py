import logging
import time

import cv2
import keras
import numpy as np
import pytesseract
import tensorflow as tf
from keras.optimizers import SGD
from PIL import Image, ImageGrab, ImageOps

from directKeys import A, D, DetectClick, PressKey, S, W, click, moveMouseTo
from Qmodels import BlockModel


def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred)


class Play2048:
    def __init__(self, logger, learning_rate=0.1, discount=0.95, exploration_rate=1.0, iterations=10000, resume=False):
        self.learning_rate = learning_rate
        self.discount = discount  # How much we appreciate future reward over current
        self.exploration_rate = 1.0  # Initial exploration rate
        # Shift from exploration to explotation
        self.exploration_delta = 1.0 / iterations
        self.max_iterations = iterations
        self.logger = logger
        self.logger.info('Initializing agent...')

        # possible moves
        self.moves = [W, A, S, D]
        self.print_moves = ['W', 'A', 'S', 'D']

        # initialize variables
        self.score = 0
        self.game_over = False
        self.old_state = []
        self.error_restart = False
        self.counter = 0

        # some window coordinates
        self.replay_coords = [1920, 1325]
        self.restart_coords = [[2155, 1765], [2002, 1656]]
        self.mmouse_coords = [763, 26]
        self.select_coords = [2045, 37]
        self.board_coords = [1471, 400, 2707, 1640]
        self.score_coords = [1824, 202, 1995, 245]

        # Output is 4 channels for W,A,S,D
        self.output_chan = 4

        # Input dimensions
        self.input_shape = (256, 256, 3)

        # model weights
        self.weights_path = 'Play2048_weights.h5'

        # build model
        self.build_model()
        if resume:
            self.model.load_weights(self.weights_path)

        self.logger.info('Agent initialized')

    def replay_click(self):
        click(self.replay_coords[0], self.replay_coords[1])

    def restart_click(self):
        click(self.restart_coords[0][0], self.restart_coords[0][1])
        time.sleep(.2)
        click(self.restart_coords[1][0], self.restart_coords[1][1])
        time.sleep(.2)

    def window_click(self):
        click(self.select_coords[0], self.select_coords[1])

    def move_mouse(self):
        moveMouseTo(self.mmouse_coords[0], self.mmoues_coords[1])

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
            self.logger.info('Game over detected')
            self.game_over = True
            return None
        config_str = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789'
        score = pytesseract.image_to_string(img_inv,
                                            config=config_str)
        self.logger.info('Got score "{}"'.format(score))
        try:
            score = np.int(score)
        except Exception as e:
            self.logger.error(e)
            self.logger.warning('Score received was: "{}"'.format(score))
            self.error_restart = True
            return None
        return score

    # Define tensorflow model graph
    def build_model(self):
        self.model = BlockModel(self.input_shape, self.output_chan)
        self.model.compile(SGD(lr=self.learning_rate), loss=huber_loss)
        self.model._make_predict_function()
        self.logger.info('Q Model created')

    # Countdown function
    def countdown(self, countdown=3):
        for count in range(countdown, 0, -1):
            self.logger.info('COUNTDOWN: {}...'.format(count))
            time.sleep(1)

    # Ask model to estimate Q value for specific state (inference)
    def get_Q(self, state):
        # Model output: Array of Q values for single state
        q_pred = self.model.predict_on_batch(state)[0]
        return q_pred

    def get_next_action(self, state):
        if np.random.rand() > self.exploration_rate:  # Explore (gamble) or exploit (greedy)
            self.logger.info('Using model action')
            action = self.greedy_action(state)
        else:
            self.logger.info('Using random action')
            action = self.random_action()
        return action

    # Which action has bigger Q-value, estimated by our model (inference).
    def greedy_action(self, state):
        # argmax picks the higher Q-value and returns the index (0,1,2,3 == W,A,S,D)
        return np.argmax(self.get_Q(state))

    def random_action(self):
        choice = np.random.randint(len(self.moves))
        return choice

    def take_action(self, action):
        PressKey(self.moves[action])
        self.logger.info('Taking action {}'.format(self.print_moves[action]))

    def img_to_input(self, img):
        array = np.array(img)
        array_rs = cv2.resize(array, self.input_shape[:2])
        inp = array_rs[np.newaxis, ...].astype(np.float)
        inp /= inp.max()
        return inp

    def get_state(self):
        state = self.get_board()
        return self.img_to_input(state)

    def train(self, old_state, action, reward, new_state):
        # Ask the model for the Q values of the old state (inference)
        old_state_Q_values = self.get_Q(old_state)

        # Ask the model for the Q values of the new state (inference)
        new_state_Q_values = self.get_Q(new_state)

        # Real Q value for the action we took. This is what we will train towards.
        old_state_Q_values[action] = reward + \
            self.discount * np.amax(new_state_Q_values)

        # Setup training data
        target = old_state_Q_values[np.newaxis, ...]

        # Train
        self.model.train_on_batch(old_state, target)

    def take_turn(self):
        # get move
        action = self.get_next_action(self.old_state)
        # make move
        self.take_action(action)
        # get new state
        new_state = self.get_state()
        # get new score
        new_score = self.get_score()
        # check for game over or error
        if new_score is None:
            self.logger.info('Ending game...')
            return
        # calculate reward
        reward = new_score - self.score
        # make reward negative if score not increased
        if reward == 0:
            reward = -10
        # Train our model with new data
        self.train(self.old_state, action, reward, new_state)

        # update old state and score
        self.old_state = new_state
        self.score = new_score

        # Shift exploration_rate toward zero (less random)
        if self.exploration_rate > 0:
            self.exploration_rate -= self.exploration_delta

    def start_game(self):
        self.score = 0
        self.game_over = False
        # Get starting state
        self.old_state = self.get_state()

        while not self.game_over and not self.error_restart:
            self.take_turn()
            self.counter += 1
            if counter == self.max_iterations:
                self.logger.info('Max iterations reached! Ending game.')
                self.game_over = True
            if counter % 50 == 0:
                self.logger.info(
                    'Saving model weights at iteration {}/{}'.format(self.counter, self.max_iterations))
                self.model.save_weights(self.weights_path)

        # If stopped because of error
        if self.error_restart:
            self.restart()
        # If stopped because game over but still iterations left
        if counter < self.max_iterations:
            self.replay()

    def replay(self):
        self.logger.info('Starting new game...')
        time.sleep(2)
        self.replay_click()
        self.move_mouse()
        time.sleep(.5)
        self.start_game()

    def restart(self):
        self.logger.info('Restarting game...')
        time.sleep(2)
        self.restart_click()
        self.move_mouse()
        time.sleep(.5)
        self.start_game()


if __name__ == '__main__':
    # setup logging
    logFormat = "%(levelname)s [%(name)s.%(funcName)s:%(lineno)d]:%(message)s"
    logging.basicConfig(level=logging.DEBUG, format=logFormat)
    logger = logging.getLogger('Play2048')
    # make file handler
    fh = logging.FileHandler('Play2048.log')
    formatter = "%(levelname)s %(asctime)s [%(name)s.%(funcName)s:%(lineno)d]:%(message)s"
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    # make console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Create agent
    agent = Play2048(logger, iterations=1000, resume=True)
    # Start countdown
    agent.countdown(3)

    # Start playing
    agent.start_game()
