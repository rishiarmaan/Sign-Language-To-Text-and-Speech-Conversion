import os

file_path = r"c:\Users\armaa\Sign-Language-To-Text-and-Speech-Conversion-master\modern_pred.py"
with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

# Replace fonts
code = code.replace('"Courier"', '"Segoe UI"')
code = code.replace("font=('Courier'", "font=('Segoe UI'")

# Replace colors to sleek modern ones
code = code.replace('#2b2b2b', '#18181B') # Right frame bg (Tailwind Zinc-900)
code = code.replace('#1f1f1f', '#27272A') # Sentence bg (Tailwind Zinc-800)
code = code.replace('#00ffcc', '#38BDF8') # Highlight color (Tailwind Sky-400)

# Replace the drawing logic dynamically
old_draw_start = "# We draw the green skeleton over the display_frame directly for AR"
old_draw_end = "cv2.circle(display_frame, (pts[i][0], pts[i][1]), 4, (0, 255, 0), cv2.FILLED)"

if old_draw_start in code and old_draw_end in code:
    start_idx = code.find(old_draw_start)
    end_idx = code.find(old_draw_end) + len(old_draw_end)
    
    new_draw = """# Sleek modern glowing overlay
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
                    cv2.putText(display_frame, f"{self.current_symbol}", (x, max(0, y - 15)), cv2.FONT_HERSHEY_DUPLEX, 1.2, (250, 150, 0), 2)"""
                    
    code = code[:start_idx] + new_draw + code[end_idx:]

with open(file_path, "w", encoding="utf-8") as f:
    f.write(code)

print("Applied UI Beautifications")
