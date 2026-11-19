import pygame, sys
from pygame.locals import *

class EventHandler:

    def __init__(self, controller):
        self.controller = controller
        self.first_key_press = False

    def handle(self, event_list):
        for event in event_list:
            if event.type == QUIT:
                quit()
            if event.type == KEYDOWN:
                self.keyboard_controller_down(event)
                self.first_key_press = True
            if event.type == KEYUP:
                self.keyboard_controller_up(event)
            if event.type == MOUSEBUTTONDOWN:
                self.mouse_down()
            if event.type == MOUSEBUTTONUP:
                self.mouse_up()

    def keyboard_controller_down(self, event):
        if event.key in (273, 1073741906, 119):
            self.controller.set_up(True)
        elif event.key in (276, 1073741904, 97):
            self.controller.set_left(True)
        elif event.key in (275, 1073741903, 100):
            self.controller.set_right(True)
        elif event.key in (113, 27):
            self.quit()

    def keyboard_controller_up(self, event):        
        if event.key in (273, 1073741906, 119):            
            self.controller.set_up(False)
        if event.key in (276, 1073741904, 97):
            self.controller.set_left(False)
        if event.key in (275, 1073741903, 100):
            self.controller.set_right(False)

    def quit(self):
        pygame.quit()
        sys.exit()

    def mouse_down(self):
        self.controller.set_mouse_pos(pygame.mouse.get_pos())
        self.controller.set_mouse(True)

    def mouse_up(self):
        self.controller.set_mouse(False)