import threading
import queue
import time
import math
import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
import pyttsx3
import enchant
from string import ascii_uppercase
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import traceback
import sys
import os

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

ddd = enchant.Dict("en-US")
offset = 29

class ModernApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Modern Sign Language Translator")
        self.root.geometry("1400x800")
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)
        
        # Grid layout
        self.root.grid_columnconfigure((0, 1), weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Left Frame (Video Feed)
        self.left_frame = ctk.CTkFrame(self.root, corner_radius=15)
        self.left_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.title_label = ctk.CTkLabel(self.left_frame, text="Real-Time Detection", font=("Segoe UI", 24, "bold"))
        self.title_label.pack(pady=10)
        
        self.video_label = ctk.CTkLabel(self.left_frame, text="")
        self.video_label.pack(expand=True)
        
        # Right Frame (Results & Info)
        self.right_frame = ctk.CTkFrame(self.root, corner_radius=15, fg_color="#18181B")
        self.right_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.current_char_label = ctk.CTkLabel(self.right_frame, text="Current Symbol: -", font=("Segoe UI", 36, "bold"), text_color="#38BDF8")
        self.current_char_label.pack(pady=30)
        
        # sentence frame
        self.sentence_frame = ctk.CTkFrame(self.right_frame, corner_radius=10, fg_color="#27272A")
        self.sentence_frame.pack(pady=20, padx=20, fill="x")
        self.sentence_label = ctk.CTkLabel(self.sentence_frame, text="Sentence:", font=("Segoe UI", 20, "bold"))
        self.sentence_label.pack(pady=5)
        self.sentence_text = ctk.CTkLabel(self.sentence_frame, text=" ", font=("Segoe UI", 30), wraplength=500, justify="left")
        self.sentence_text.pack(pady=20)
        
        # Top 3 Confidence Meters
        self.conf_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.conf_frame.pack(pady=20, padx=20, fill="x")
        
        self.conf_title = ctk.CTkLabel(self.conf_frame, text="Confidence", font=("Segoe UI", 20, "bold"))
        self.conf_title.grid(row=0, column=0, columnspan=2, pady=10)
        
        self.conf_labels = []
        self.conf_bars = []
        for i in range(3):
            lbl = ctk.CTkLabel(self.conf_frame, text=f"Rank {i+1}", font=("Segoe UI", 16))
            lbl.grid(row=i+1, column=0, padx=10, pady=5, sticky="e")
            bar = ctk.CTkProgressBar(self.conf_frame, width=200)
            bar.grid(row=i+1, column=1, padx=10, pady=5, sticky="w")
            bar.set(0)
            self.conf_labels.append(lbl)
            self.conf_bars.append(bar)
            
        # Suggestions buttons
        self.sugg_label = ctk.CTkLabel(self.right_frame, text="Suggestions", font=("Segoe UI", 20, "bold"))
        self.sugg_label.pack(pady=10)
        
        self.sugg_buttons_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.sugg_buttons_frame.pack(pady=10)
        self.sugg_buttons = []
        for i in range(4):
            btn = ctk.CTkButton(self.sugg_buttons_frame, text="-", font=("Segoe UI", 16), width=100, command=lambda idx=i: self.apply_suggestion(idx))
            btn.grid(row=0, column=i, padx=5)
            self.sugg_buttons.append(btn)
            
        # Bottom controls
        self.controls_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.controls_frame.pack(side="bottom", pady=30)
        
        self.space_btn = ctk.CTkButton(self.controls_frame, text="Space", width=80, command=self.btn_space)
        self.space_btn.grid(row=0, column=0, padx=5)
        
        self.backspace_btn = ctk.CTkButton(self.controls_frame, text="Backspace", width=80, command=self.btn_backspace)
        self.backspace_btn.grid(row=0, column=1, padx=5)
        
        self.next_btn = ctk.CTkButton(self.controls_frame, text="Next", width=80, command=self.btn_next)
        self.next_btn.grid(row=0, column=2, padx=5)
        
        self.clear_btn = ctk.CTkButton(self.controls_frame, text="Clear", width=80, fg_color="#d9534f", hover_color="#c9302c", command=self.clear_text)
        self.clear_btn.grid(row=0, column=3, padx=5)
        
        self.speak_btn = ctk.CTkButton(self.controls_frame, text="Speak", width=80, fg_color="#5cb85c", hover_color="#449d44", command=self.speak_text)
        self.speak_btn.grid(row=0, column=4, padx=5)
        
        self.q = queue.Queue()
        self.running = True
        
        # Engine stuff
        self.model = load_model('cnn8grps_rad1_model.h5')
        self.hd = HandDetector(maxHands=1)
        self.hd2 = HandDetector(maxHands=1)
        self.speak_engine = pyttsx3.init()
        self.speak_engine.setProperty("rate", 120)
        
        self.str = " "
        self.word = " "
        self.ccc = 0
        self.current_symbol = "C"
        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = [" "] * 10
        self.pred_buffer = []

        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "
        
        self.top3_probs = []
        
        # Start AI thread
        self.thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.thread.start()
        
        # Start UI loop
        self.update_ui()
        
    def camera_loop(self):
        vs = cv2.VideoCapture(0)
        while self.running:
            ok, frame = vs.read()
            if not ok:
                time.sleep(0.01)
                continue
            
            # Flip frame natively
            frame = cv2.flip(frame, 1)
            
            # Detect hand and draw on the LIVE frame (AR overlay style)
            # Find hands and draw bounding box + skeleton on screen natively
            hands = self.hd.findHands(frame, draw=False, flipType=True)
            
            display_frame = frame.copy()
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                
                # Sleek modern glowing overlay
                overlay = display_frame.copy()
                pts = hand['lmList']
                
                # Draw sleek translucent connections
                connections = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (5,9), (9,10), (10,11), (11,12), (9,13), (13,14), (14,15), (15,16), (13,17), (0,17), (17,18), (18,19), (19,20)]
                for c1, c2 in connections:
                    cv2.line(overlay, (pts[c1][0], pts[c1][1]), (pts[c2][0], pts[c2][1]), (250, 150, 0), 2)
                    
                # Add outer glow
                for i in range(21):
                    cv2.circle(overlay, (pts[i][0], pts[i][1]), 8, (250, 150, 0), -1)
                
                # Alpha blending
                cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)
                
                # Solid center cores to complete the Iron-Man HUD effect
                for i in range(21):
                    cv2.circle(display_frame, (pts[i][0], pts[i][1]), 3, (255, 255, 255), -1)
                
                # Modern target box with corners
                import cvzone
                cvzone.cornerRect(display_frame, [x, y, w, h], colorC=(250, 150, 0), colorR=(0, 255, 255), t=3, rt=1)
                
                # Draw dynamic text near hand
                if hasattr(self, 'current_symbol') and self.current_symbol:
                    cv2.putText(display_frame, f"{self.current_symbol}", (x, max(0, y - 15)), cv2.FONT_HERSHEY_DUPLEX, 1.2, (250, 150, 0), 2)
                
                # Now extract crop to feed to the network
                try:
                    image = frame[max(0, y - offset):y + h + offset, max(0, x - offset):x + w + offset]
                    white = np.ones((400,400,3), dtype=np.uint8) * 255
                    if image is not None and image.size != 0:
                        handz = self.hd2.findHands(image, draw=False, flipType=True)
                        if handz:
                            hand_white = handz[0]
                            self.pts = hand_white['lmList']
                            os_x = ((400 - w) // 2) - 15
                            os_y = ((400 - h) // 2) - 15
                            for t in range(0, 4, 1):
                                cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                            for t in range(5, 8, 1):
                                cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                            for t in range(9, 12, 1):
                                cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                            for t in range(13, 16, 1):
                                cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                            for t in range(17, 20, 1):
                                cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                            cv2.line(white, (self.pts[5][0] + os_x, self.pts[5][1] + os_y), (self.pts[9][0] + os_x, self.pts[9][1] + os_y), (0, 255, 0), 3)
                            cv2.line(white, (self.pts[9][0] + os_x, self.pts[9][1] + os_y), (self.pts[13][0] + os_x, self.pts[13][1] + os_y), (0, 255, 0), 3)
                            cv2.line(white, (self.pts[13][0] + os_x, self.pts[13][1] + os_y), (self.pts[17][0] + os_x, self.pts[17][1] + os_y), (0, 255, 0), 3)
                            cv2.line(white, (self.pts[0][0] + os_x, self.pts[0][1] + os_y), (self.pts[5][0] + os_x, self.pts[5][1] + os_y), (0, 255, 0), 3)
                            cv2.line(white, (self.pts[0][0] + os_x, self.pts[0][1] + os_y), (self.pts[17][0] + os_x, self.pts[17][1] + os_y), (0, 255, 0), 3)

                            for i in range(21):
                                cv2.circle(white, (self.pts[i][0] + os_x, self.pts[i][1] + os_y), 2, (0, 255, 0), 1)
                            
                            self.predict(white)
                except Exception as e:
                    print("Error parsing hand", e)
                    pass

            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(display_frame)
            
            # Use put to send to UI thread safely
            if not self.q.full():
                self.q.put((img, self.current_symbol, self.str, [self.word1, self.word2, self.word3, self.word4], self.top3_probs))
                
            time.sleep(0.01)
        vs.release()

    def update_ui(self):
        try:
            while not self.q.empty():
                img, sym, sentence, suggs, probes = self.q.get_nowait()
                imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
                self.video_label.configure(image=imgtk)
                if sym:
                    self.current_char_label.configure(text=f"Current Symbol: {sym}")
                self.sentence_text.configure(text=sentence)
                
                for i in range(4):
                    self.sugg_buttons[i].configure(text=suggs[i] if suggs[i].strip() else "-")
                    
                if len(probes) >= 3:
                     # e.g., probes = [(char1, prob1), ...]
                     for i in range(3):
                         self.conf_labels[i].configure(text=str(probes[i][0]))
                         self.conf_bars[i].set(probes[i][1])

        except Exception as e:
            pass
        self.root.after(15, self.update_ui)
        
    def distance(self,x,y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def apply_suggestion(self, idx):
        words = [self.word1, self.word2, self.word3, self.word4]
        target = words[idx]
        if not target.strip(): return
        
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, max(0, idx_space))
        if idx_word == -1: idx_word = max(0, idx_space)
        self.str = self.str[:idx_word]
        self.str = self.str + target.upper() + " "

    def speak_text(self):
        def _speak():
            self.speak_engine.say(self.str)
            self.speak_engine.runAndWait()
        threading.Thread(target=_speak, daemon=True).start()

    def btn_space(self):
        self.str += " "

    def btn_backspace(self):
        if len(self.str) > 0 and self.str != " ":
            self.str = self.str.rstrip(" ")
            self.str = self.str[:-1]
        if len(self.str) == 0:
            self.str = " "

    def btn_next(self):
        if self.current_symbol and self.current_symbol not in ["", -1, "next", " ", "Backspace"]:
            self.str += str(self.current_symbol)

    def clear_text(self):
        self.str = " "
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

    def on_close(self):
        self.running = False
        self.root.destroy()
        
    def predict(self, test_image):
        white=test_image
        white = white.reshape(1, 400, 400, 3)
        raw_prob = self.model.predict(white, verbose=0)[0]
        prob = np.array(raw_prob, dtype='float32')
        
        max_prob = np.max(prob)
        ch1 = np.argmax(prob, axis=0)

        # Confidence Thresholding
        if max_prob < 0.5:
            ch1 = -1 # Special tag for unconfident

        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0

        pl = [ch1, ch2]

        # condition for [Aemnst]
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 0

        # condition for [o][s]
        l = [[2, 2], [2, 1]]
        if pl in l:
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0
                print("++++++++++++++++++")
                # print("00000")

        # condition for [c0][aemnst]
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][
                0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2

        # condition for [c0][aemnst]
        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2


        # condition for [gh][bdfikruvw]
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]

        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][
                0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 3



        # con for [gh][l]
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3

        # con for [gh][pqz]
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3

        # con for [l][x]
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4

        # con for [l][d]
        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) > 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 4

        # con for [l][gh]
        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[0][0]):
                ch1 = 4

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        # con for [gh][z]
        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] > self.pts[10][1]:
                ch1 = 5

        # con for [gh][pq]
        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][
                1] + 17 > self.pts[20][1]:
                ch1 = 5

        # con for [l][pqz]
        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 5

        # con for [pqz][aemnst]
        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 5

        # con for [pqz][yj]
        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[3][0] < self.pts[0][0]:
                ch1 = 7

        # con for [l][yj]
        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] < self.pts[8][1]:
                ch1 = 7

        # con for [x][yj]
        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] > self.pts[20][1]:
                ch1 = 7

        # condition for [x][aemnst]
        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] > self.pts[16][0]:
                ch1 = 6


        # condition for [yj][x]
        print("2222  ch1=+++++++++++++++++", ch1, ",", ch2)
        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                ch1 = 6

        # condition for [c0][x]
        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) > 50:
                ch1 = 6

        # con for [l][x]

        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) < 60:
                ch1 = 6

        # con for [x][d]
        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                ch1 = 6

        # con for [b][pqz]
        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
             [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 1

        # con for [f][pqz]
        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
             [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        # con for [d][pqz]
        fg = 19
        # print("_________________ch1=",ch1," ch2=",ch2)
        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                ch1 = 1

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) < 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 1

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                ch1 = 1

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                ch1 = 1

        # con for [i][pqz]
        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] > self.pts[20][1])):
                ch1 = 1

        # con for [yj][bfdi]
        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[5][0] + 15) and (
            (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
             self.pts[18][1] > self.pts[20][1])):
                ch1 = 7

        # con for [uvr]
        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1

        # con for [w]
        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][0] + fg < self.pts[16][0] and
                    self.pts[0][0] + fg < self.pts[20][0]) and not (
                    self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][
                0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                ch1 = 1

        # con for [w]

        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                ch1 = 1

        # -------------------------condn for 8 groups  ends

        # -------------------------condn for subgroups  starts
        #
        if ch1 == 0:
            ch1 = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                ch1 = 'A'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][
                0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'T'
            if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                ch1 = 'E'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'M'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                ch1 = 'N'

        if ch1 == 2:
            if self.distance(self.pts[12], self.pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if (self.distance(self.pts[8], self.pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if self.distance(self.pts[8], self.pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'

        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                if self.pts[8][1] < self.pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'B'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'D'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'F'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'I'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'W'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'U'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'

            if (self.pts[8][0] > self.pts[12][0]) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'R'

        if ch1 in [1, 'E', 'S', 'X', 'Y', 'B']:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1=" "

        if ch1 in ['E', 'Y', 'B']:
            if (self.pts[4][0] < self.pts[5][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1="next"

        if ch1 in ['Next', 'B', 'C', 'H', 'F', 'X']:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                ch1 = 'Backspace'

        # Temporal Smoothing
        if not hasattr(self, 'pred_buffer'): self.pred_buffer = []
        if ch1 != -1: # Ignore completely unconfident frames
            self.pred_buffer.append(ch1)
        if len(self.pred_buffer) > 10: self.pred_buffer.pop(0)

        import collections
        if self.pred_buffer:
            mode_ch1 = collections.Counter(self.pred_buffer).most_common(1)[0][0]
            ch1 = mode_ch1

        if ch1 == "next" and self.prev_char != "next":
            target_char = " "
            for c in reversed(self.pred_buffer[:-1]):
                if c not in ["next", " ", "", -1]:
                    target_char = c
                    break

            if target_char == "Backspace":
                if len(self.str) > 0 and self.str != " ":
                    self.str = self.str.rstrip(" ")
                    self.str = self.str[:-1]
                if len(self.str) == 0:
                    self.str = " "
            elif target_char != " ":
                self.str = self.str + str(target_char)

        if ch1 == "  " and self.prev_char != "  ":
            self.str = self.str + " "

        self.prev_char = ch1
        self.current_symbol = ch1 if ch1 != -1 else ""
        self.count += 1
        self.ten_prev_char[self.count%10] = ch1


        if len(self.str.strip())!=0:
            st=self.str.rfind(" ")
            ed=len(self.str)
            word=self.str[st+1:ed]
            self.word=word
            if len(word.strip())!=0:
                ddd.check(word)
                lenn = len(ddd.suggest(word))
                if lenn >= 4:
                    self.word4 = ddd.suggest(word)[3]

                if lenn >= 3:
                    self.word3 = ddd.suggest(word)[2]

                if lenn >= 2:
                    self.word2 = ddd.suggest(word)[1]

                if lenn >= 1:
                    self.word1 = ddd.suggest(word)[0]
            else:
                self.word1 = " "
                self.word2 = " "
                self.word3 = " "
                self.word4 = " "




if __name__ == "__main__":
    root = ctk.CTk()
    app = ModernApp(root)
    root.mainloop()

