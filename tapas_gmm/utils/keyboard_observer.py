import time
from functools import partial

import numpy as np
from loguru import logger

try:
    from pynput import keyboard
except ImportError:
    keyboard = None
    logger.info("Running headless. Better don't touch the keyboard.")


class KeyboardObserver:
    def __init__(self):
        self.reset()
        self.hotkeys = keyboard.GlobalHotKeys(
            {
                "g": partial(self.set_label, 1),  # good
                "b": partial(self.set_label, 0),  # bad
                "c": partial(self.set_gripper, -0.9),  # close
                "v": partial(self.set_gripper, 0.9),  # open
                "f": partial(self.set_gripper, None),  # gripper free
                "x": partial(self.reset_episode, True),
                "y": partial(self.reset_episode, False),
            }
        )
        self.hotkeys.start()
        self.direction = np.array([0, 0, 0, 0, 0, 0])
        self.listener = keyboard.Listener(
            on_press=self.set_direction, on_release=self.reset_direction
        )
        self.key_mapping = {
            "a": (1, 1),  # left
            "d": (1, -1),  # right
            "s": (0, 1),  # backward
            "w": (0, -1),  # forward
            "q": (2, 1),  # down
            "e": (2, -1),  # up
            "j": (3, -1),  # look left
            "l": (3, 1),  # look right
            "i": (4, -1),  # look up
            "k": (4, 1),  # look down
            "u": (5, -1),  # rotate left
            "o": (5, 1),  # rotate right
        }
        self.listener.start()
        # self.listener.join()
        # self.current_modifiers = set()

    def set_label(self, value):
        self.label = value
        logger.info("label set to: ", value)

    def get_label(self):
        return self.label

    def set_gripper(self, value):
        self.gripper_open = value
        logger.info("gripper set to: ", value)

    def get_gripper(self):
        return self.gripper_open

    def set_direction(self, key):
        # self.current_modifiers.add(key)
        # if all('c' in self.current_modifiers and keyboard.Key.ctrl in self.current_modifiers):
        #     raise KeyboardInterrupt
        try:
            idx, value = self.key_mapping[key.char]
            self.direction[idx] = value
        except (KeyError, AttributeError):
            pass

    def reset_direction(self, key):
        # self.current_modifiers.remove(key)
        try:
            idx, _ = self.key_mapping[key.char]
            self.direction[idx] = 0
        except (KeyError, AttributeError):
            pass

    def has_joints_cor(self):
        return self.direction.any()

    def has_gripper_update(self):
        return self.get_gripper() is not None

    def get_ee_action(self):
        return self.direction * 0.9

    def reset_episode(self, success):
        self.reset_button = True
        self.success = success

    def reset(self):
        self.set_label(1)
        self.set_gripper(None)
        self.reset_button = False
        self.success = None


@logger.contextualize(filter=False)
def wait_for_environment_reset(env, keyboard_obs):
    if keyboard_obs is not None:
        env.reset()
        logger.info("Waiting for env reset. Confirm via input ...")
        while True:
            env.get_obs(update_visualization=True)
            time.sleep(0.5)
            if keyboard_obs.reset_button or keyboard_obs.success:
                keyboard_obs.reset()
                break
