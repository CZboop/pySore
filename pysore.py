import cv2
import random
import numpy as np
import PIL
from PIL import Image
import matplotlib
from matplotlib import cm
import threading
from functools import partial
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.uix.button import MDRaisedButton
from kivy.uix.widget import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.graphics import Color
from kivy.graphics import Rectangle
from kivy.core.window import Window
from kivy.uix.colorpicker import ColorPicker
from kivymd.app import MDApp
from kivymd.uix.label import MDLabel, MDIcon
from datetime import datetime

class MainScreen(Screen):
    pass

class MenuScreen(Screen):
    pass

class Manager(ScreenManager):
    pass

Window.size = (300,500)

builder_str = '''
ScreenManager:
    MenuScreen:
    MainScreen:

<MenuScreen>:
    name: 'Menu'
    FloatLayout:
        MDLabel:
            text: "pySore"
            pos_hint: {'center_x':.5, 'center_y':.7}
            size_hint: 1.0, 0.2
            font_style: 'H3'
            color: (1,1,1,1)
            halign: 'center'

        MDLabel:
            text: "Eyesore filters \\nmade with Python!"
            pos_hint: {'center_x':.5, 'center_y':.4}
            size_hint: 1.0, 0.15
            font_style: 'H6'
            color: (1,1,1,1)
            halign: 'center'

        MDIcon:
            icon: 'eye'
            pos_hint: {'center_x':.25, 'center_y':.55}
            halign: 'center'

        MDIcon:
            icon: 'language-python'
            pos_hint: {'center_x':.5, 'center_y':.55}
            halign: 'center'
        MDIcon:
            icon: 'camera-party-mode'
            pos_hint: {'center_x':.75, 'center_y':.55}
            halign: 'center'

        MDRaisedButton:
            text: 'Begin!'
            size_hint: 0.6, 0.1
            pos_hint: {'center_x': .5, 'center_y': .2}
            font_style: 'H5'
            on_release:
                root.manager.transition.direction='left'
                root.manager.current = 'Main'

<MainScreen>:
    name: "Main"
    FloatLayout:
        Image:
            id: vid
            size_hint: 1, 0.6
            allow_stretch: True
            keep_ratio: True
            pos_hint: {'center_x':0.5, 'top':0.8}

        MDRaisedButton:
            text: 'Save Image'
            pos_hint: {"x":0.0, "y":0.0}
            size_hint: 1.0, 0.12
            font_style: 'H6'
            on_release: app.save_img()

        MDToolbar:
            id: toolbar
            title: 'Menu'
            pos_hint: {'top': 1}
            elevation: 15
            left_action_items: [["menu", lambda x: nav_drawer.set_state("toggle")]]
        Widget:

    MDNavigationDrawer:
        id: nav_drawer
        FloatLayout:
            size_hint: 1.0, 1.0
            MDLabel:
                text: "Try Out A Filter!"
                size_hint: 1.0, 0.1
                pos_hint: {"x":0.3, "y":0.9}
            MDRaisedButton:
                text: "Lizard Queen"
                size_hint: 1.0, 0.09
                pos_hint: {"x":0.0, "y":0.81}
                on_release: app.set_state("lizard")
            MDRaisedButton:
                text: "ASCII Fine"
                size_hint: 1.0, 0.09
                pos_hint: {"x":0.0, "y":0.72}
                on_release: app.set_state("ascii")
            MDRaisedButton:
                text: "Crying Pixels"
                size_hint: 1.0, 0.09
                pos_hint: {"x":0.0, "y":0.63}
                on_release: app.set_state("cry_glitch")
            MDRaisedButton:
                text: "Scanline Party"
                size_hint: 1.0, 0.09
                pos_hint: {"x":0.0, "y":0.54}
                on_release: app.set_state("scanline_party")
            MDRaisedButton:
                text: "Pixel Sort"
                size_hint: 1.0, 0.09
                pos_hint: {"x":0.0, "y":0.45}
                on_release: app.set_state("pix_sort")
            MDRaisedButton:
                text: "Scanlines"
                size_hint: 1.0, 0.09
                pos_hint: {"x":0.0, "y":0.36}
                on_release: app.set_state("scanlines")
            MDRaisedButton:
                text: "ASCII Chunky"
                size_hint: 1.0, 0.09
                pos_hint: {"x":0.0, "y":0.27}
                on_release: app.set_state("ascii_small")
            MDRaisedButton:
                text: "Night Vision"
                size_hint: 1.0, 0.09
                pos_hint: {"x":0.0, "y":0.18}
                on_release: app.set_state("night_vision")
            MDRaisedButton:
                text: "Deep Fried"
                size_hint: 1.0, 0.09
                pos_hint: {"x":0.0, "y":0.09}
                on_release: app.set_state("fried")
            MDRaisedButton:
                text: "No Filter"
                size_hint: 1.0, 0.09
                pos_hint: {"x":0.0, "y":0.0}
                on_release: app.set_state("")

'''

class pySore(MDApp):
    # properties to be assigned later - state will be hold the current filter, frame is the camera feed
    state = ""
    frame = None

    # building the app with the .kv string above and screen class instances for each screen
    def build(self):
        # setting some colour themes
        self.theme_cls.primary_palette = "DeepPurple"
        self.theme_cls.theme_style = "Dark"

        # using the builder to parse the .kv string above for use by the app
        Builder.load_string(builder_str)

        # setting up a thread so the camera feed will be accessible within the app
        threading.Thread(target=self.feed_video, daemon=True).start()

        # adding screens to the screen manager
        sm = ScreenManager()
        self.welcome_screen = MenuScreen()
        sm.add_widget(self.welcome_screen)
        self.main_screen = MainScreen()
        sm.add_widget(self.main_screen)

        #returning the screen manager, now complete with all screens
        return sm

    # main method to display the video and distort based on filters chosen
    def feed_video(self):
        # boolean can be changed to stop the video feed, currently just stays true
        self.video_live = True

        # setting primary camera as input
        cam = cv2.VideoCapture(0)

        # main video loop
        while (self.video_live):

            # loading haar cascades for face and eye detection for use in some filters
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

            # getting video input
            ret, frame = cam.read()

            # using if statements for each filter
            # scanlines filter simply overlays thin black lines down the frame
            if self.state == "scanlines":
                line = []
                pix_line = [[0,0,0]]*frame.shape[1]
                for i in range(1):
                    line.append(pix_line)
                for i in range(0, frame.shape[0], 4):
                    frame[i:i+2, :] = line

            # a variation of the main scanline filter, uses it as if it was a translucent overlay, toggles between random colours
            if self.state == "scanline_party":
                line = []
                pix_line = [[random.randrange(0,256),random.randrange(0,256),random.randrange(0,256)]]*frame.shape[1]
                for i in range(2):
                    line.append(pix_line)
                for i in range(0, frame.shape[0], 4):
                    frame[i:i+2, :] = line

            # messing with the numpy array to create a cool, jagged rainbow effect
            if self.state == "lizard":
                frame = np.sort(frame, axis=0)[::-1]**2

            # a vertical pixel sort across the whole frame
            if self.state == "pix_sort":
                frame = np.sort(frame, axis=0)[::-1]

            # fine/ small font ascii art creates a smooth silver effect
            if self.state == "ascii":
                    ascii_list = ["@","&", "#", "%", "?", ";",":","*",",","."]
                    # reducing resolution so each pixel can be a character, and so frame rate can be maintained
                    frame =  cv2.resize(frame, (60,60))
                    # change to greyscale
                    frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    #rotating the frame as a later operation rotates it in the inverse direction
                    frame = np.rot90(frame)
                    # looping through pixels of the frame, dividing by 26 so the colour value can be used as an index in ascii_list
                    ascii_pixels = []
                    for x in frame:
                        for y in x:
                            # adding the character with corresponding intensity/darkness to the pixel list
                            ascii_pixels.append(ascii_list[y//26])
                    # creating a black background
                    frame = np.zeros((1512,1512,3), np.uint8)
                    # turning the flat ascii character array into a 2d array of size of the reduced resolution input
                    ascii_2d = []
                    for i in range(60):
                        ascii_2d.append(ascii_pixels[i*60:i*60+60])
                    # using the two indices of the array to add the corresponding corresponding character in the correct place
                    for c,v in enumerate(ascii_2d):
                        for i,j in enumerate(v):
                            cv2.putText(frame, j, (c*25,i*25), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1, cv2.LINE_AA)

            # the same as above, with lower original resolution, less characters overall that are larger and more spaced out
            # looks more like traditional ascii art
            if self.state == "ascii_small":
                    ascii_list = ["@","&", "#", "%", "?", ";",":","*",",","."]
                    frame =  cv2.resize(frame, (30,30))
                    frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = np.rot90(frame)

                    ascii_pixels = []
                    for x in frame:
                        for y in x:
                            ascii_pixels.append(ascii_list[y//26])

                    frame = np.zeros((1512,1512,3), np.uint8)

                    ascii_2d = []
                    for i in range(30):
                        ascii_2d.append(ascii_pixels[i*30:i*30+30])

                    for c,v in enumerate(ascii_2d):
                        for i,j in enumerate(v):
                            cv2.putText(frame, j, (c*50,i*50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # distorting the image from the eyes down, creates a glitchy 'crying' effect
            # currently jittery due to eye detection
            if self.state == "cry_glitch":
                # detecting any faces first to cut down on work for the app and improve accuracy
                faces = face_cascade.detectMultiScale(grey, 1.3, 10)
                # getting the location of faces
                for (x, y, w, h) in faces:
                    roi_grey = grey[y:y+h, x:x+w]
                    roi_colour = frame[y:y+h, x:x+w]
                    face_shape = roi_colour.shape
                    # detecting eyes within faces
                    eyes = eye_cascade.detectMultiScale(grey, 1.3, 10)
                    # iterating over list of eyes
                    for (ex, ey, ew, eh) in eyes:

                        # sorting and distorting the area from most of the way down the eyes to the bottom of frame
                        sorted_arr = -np.sort(-frame[ey+eh//3*2:, ex:ex+ew], axis=0)
                        # reassigning the relevant area in the frame to the sorted numpy array slice above
                        frame[ey+eh//3*2:, ex:ex+ew] = sorted_arr

            # negative of the frame squared creates small clumps of vivid colours
            if self.state == "fried":
                frame = -frame**2

            # negative of the frame
            if self.state == "night_vision":
                frame = -frame

            # prepping the frame to be displayed
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Clock.schedule_once(partial(self.display_frame, frame))

            #assigning the frame property and setting waitkey to show as video
            self.frame = frame;
            cv2.waitKey(1)

        # when the while condition is not met, ending the cv2 feed
        cam.release()
        cv2.destroyAllWindows()

    # currently unused function to stop the video feed
    def stop_vid(self):
        self.video_live = False

    # image processing to display the frame within the app's Image widget
    def display_frame(self, frame, dt):
        # making a texture to fit the video input
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        # assign texture to the frame value
        texture.blit_buffer(frame.tobytes(order=None), colorfmt='bgr', bufferfmt='ubyte')
        # flipping back to the right way up
        texture.flip_vertical()
        # assigning this new texture to the Image widget using it's id
        self.main_screen.ids.vid.texture = texture

    # setting the state for filter switching
    def set_state(self, new_state):
        self.state = new_state

    # saving the image using cv2 inbuilt function, using the current time to generate the filename
    def save_img(self):
        cv2.imwrite(datetime.now().strftime("%Y%m%d-%H%M%S")+ ".jpg", self.frame)

# running the app
if __name__ == '__main__':
    pySore().run()
