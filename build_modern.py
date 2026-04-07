import re

with open('final_pred.py', 'r') as f:
    text = f.read()

# Extract 'def predict(self, test_image):' ending exactly before 'def action1' or similar, but wait, `action1` etc are defined BEFORE `predict()` in final_pred.py.
# The methods after `predict` are just destructor. So `predict` goes until `def destructor(self):`.
predict_match = re.search(r'(    def predict\(self, test_image\):[\s\S]*?)(?=    def destructor\()', text)
if predict_match:
    predict_method = predict_match.group(1)
else:
    print("Could not find predict method")
    exit(1)

# Now let's assemble the new modern_pred.py
template = f"""import threading
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
        
        self.title_label = ctk.CTkLabel(self.left_frame, text="Real-Time Detection", font=("Courier", 24, "bold"))
        self.title_label.pack(pady=10)
        
        self.video_label = ctk.CTkLabel(self.left_frame, text="")
        self.video_label.pack(expand=True)
        
        # Right Frame (Results & Info)
        self.right_frame = ctk.CTkFrame(self.root, corner_radius=15, fg_color="#2b2b2b")
        self.right_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.current_char_label = ctk.CTkLabel(self.right_frame, text="Current Symbol: -", font=("Courier", 36, "bold"), text_color="#00ffcc")
        self.current_char_label.pack(pady=30)
        
        # sentence frame
        self.sentence_frame = ctk.CTkFrame(self.right_frame, corner_radius=10, fg_color="#1f1f1f")
        self.sentence_frame.pack(pady=20, padx=20, fill="x")
        self.sentence_label = ctk.CTkLabel(self.sentence_frame, text="Sentence:", font=("Courier", 20, "bold"))
        self.sentence_label.pack(pady=5)
        self.sentence_text = ctk.CTkLabel(self.sentence_frame, text=" ", font=("Courier", 30), wraplength=500, justify="left")
        self.sentence_text.pack(pady=20)
        
        # Top 3 Confidence Meters
        self.conf_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.conf_frame.pack(pady=20, padx=20, fill="x")
        
        self.conf_title = ctk.CTkLabel(self.conf_frame, text="Confidence", font=("Courier", 20, "bold"))
        self.conf_title.grid(row=0, column=0, columnspan=2, pady=10)
        
        self.conf_labels = []
        self.conf_bars = []
        for i in range(3):
            lbl = ctk.CTkLabel(self.conf_frame, text=f"Rank {{i+1}}", font=("Courier", 16))
            lbl.grid(row=i+1, column=0, padx=10, pady=5, sticky="e")
            bar = ctk.CTkProgressBar(self.conf_frame, width=200)
            bar.grid(row=i+1, column=1, padx=10, pady=5, sticky="w")
            bar.set(0)
            self.conf_labels.append(lbl)
            self.conf_bars.append(bar)
            
        # Suggestions buttons
        self.sugg_label = ctk.CTkLabel(self.right_frame, text="Suggestions", font=("Courier", 20, "bold"))
        self.sugg_label.pack(pady=10)
        
        self.sugg_buttons_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.sugg_buttons_frame.pack(pady=10)
        self.sugg_buttons = []
        for i in range(4):
            btn = ctk.CTkButton(self.sugg_buttons_frame, text="-", font=("Courier", 16), width=100, command=lambda idx=i: self.apply_suggestion(idx))
            btn.grid(row=0, column=i, padx=5)
            self.sugg_buttons.append(btn)
            
        # Bottom controls
        self.controls_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.controls_frame.pack(side="bottom", pady=30)
        
        self.clear_btn = ctk.CTkButton(self.controls_frame, text="Clear", fg_color="#d9534f", hover_color="#c9302c", command=self.clear_text)
        self.clear_btn.grid(row=0, column=0, padx=10)
        self.speak_btn = ctk.CTkButton(self.controls_frame, text="Speak", fg_color="#5cb85c", hover_color="#449d44", command=self.speak_text)
        self.speak_btn.grid(row=0, column=1, padx=10)
        
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
                
                # We draw the green skeleton over the display_frame directly for AR
                # The hands array contains lmList
                pts = hand['lmList']
                for t in range(0, 4, 1):
                    cv2.line(display_frame, (pts[t][0], pts[t][1]), (pts[t + 1][0], pts[t + 1][1]), (0, 255, 255), 3)
                for t in range(5, 8, 1):
                    cv2.line(display_frame, (pts[t][0], pts[t][1]), (pts[t + 1][0], pts[t + 1][1]), (0, 255, 255), 3)
                for t in range(9, 12, 1):
                    cv2.line(display_frame, (pts[t][0], pts[t][1]), (pts[t + 1][0], pts[t + 1][1]), (0, 255, 255), 3)
                for t in range(13, 16, 1):
                    cv2.line(display_frame, (pts[t][0], pts[t][1]), (pts[t + 1][0], pts[t + 1][1]), (0, 255, 255), 3)
                for t in range(17, 20, 1):
                    cv2.line(display_frame, (pts[t][0], pts[t][1]), (pts[t + 1][0], pts[t + 1][1]), (0, 255, 255), 3)
                cv2.line(display_frame, (pts[5][0], pts[5][1]), (pts[9][0], pts[9][1]), (0, 255, 255), 3)
                cv2.line(display_frame, (pts[9][0], pts[9][1]), (pts[13][0], pts[13][1]), (0, 255, 255), 3)
                cv2.line(display_frame, (pts[13][0], pts[13][1]), (pts[17][0], pts[17][1]), (0, 255, 255), 3)
                cv2.line(display_frame, (pts[0][0], pts[0][1]), (pts[5][0], pts[5][1]), (0, 255, 255), 3)
                cv2.line(display_frame, (pts[0][0], pts[0][1]), (pts[17][0], pts[17][1]), (0, 255, 255), 3)

                for i in range(21):
                    cv2.circle(display_frame, (pts[i][0], pts[i][1]), 4, (0, 255, 0), cv2.FILLED)
                
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
                    self.current_char_label.configure(text=f"Current Symbol: {{sym}}")
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

    def clear_text(self):
        self.str = " "
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

    def on_close(self):
        self.running = False
        self.root.destroy()
        
{predict_method}

if __name__ == "__main__":
    root = ctk.CTk()
    app = ModernApp(root)
    root.mainloop()

"""

# Let's adjust predict_method so it saves self.top3_probs.
# predict method has `prob = np.array(...)` and `ch1, ch2, ch3`. It's hard to inject exactly.
# I will do a regex replacement inside predict_method text before writing.
predict_method = predict_method.replace(
    '''        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0''',
    '''        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0
        try:
            val1 = float(raw_prob[ch1]) if max_prob >= 0.5 else 0
            val2 = float(raw_prob[ch2])
            val3 = float(raw_prob[ch3])
            # Mapping class numbers to characters, roughly. since true mapping is complex, we just show numbers as confidence labels and update them if needed. 
            self.top3_probs = [(f"Grp {{ch1}}", val1), (f"Grp {{ch2}}", val2), (f"Grp {{ch3}}", val3)]
        except: pass
'''
)

with open('modern_pred.py', 'w') as f:
    f.write(template)
print("done writing modern_pred.py")
