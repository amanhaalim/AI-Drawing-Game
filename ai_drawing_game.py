import cv2
import mediapipe as mp
import numpy as np
import requests
import json
import threading
from collections import deque
import time
import random
import base64

class AIDrawingGame:
    def __init__(self):
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Drawing state
        self.canvas = None
        self.drawing_points = deque(maxlen=3000)
        self.prev_point = None
        self.point_buffer = deque(maxlen=5)
        
        # AI state
        self.ai_guess = ""
        self.ai_processing = False
        self.ai_confidence = 0
        self.hint_text = ""
        
        # Game state
        self.score = 0
        self.streak = 0
        self.best_streak = 0
        self.total_drawings = 0
        self.correct_guesses = 0
        
        # Game modes
        self.game_mode = "FREE"  # FREE, CHALLENGE, TIME_ATTACK
        self.challenge_object = ""
        self.time_remaining = 0
        self.time_attack_start = 0
        
        # Drawing settings
        self.is_drawing_mode = False
        self.draw_colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        self.current_color_idx = 0
        self.draw_color = self.draw_colors[0]
        self.draw_thickness = 10
        
        # Achievements
        self.achievements = {
            'first_guess': False,
            'streak_3': False,
            'streak_5': False,
            'streak_10': False,
            'speed_demon': False,
            'artist': False,
            'master': False,
        }
        self.new_achievement = ""
        self.achievement_time = 0
        
        # Challenge objects list
        self.easy_objects = [
            "circle", "square", "triangle", "star", "heart",
            "sun", "moon", "house", "tree", "flower",
            "car", "boat", "ball", "box", "stick figure"
        ]
        
        self.medium_objects = [
            "cat", "dog", "fish", "bird", "butterfly",
            "mountain", "cloud", "rainbow", "umbrella", "cup"
        ]
        
        self.hard_objects = [
            "elephant", "giraffe", "rocket", "airplane", "bicycle",
            "camera", "guitar", "piano", "crown", "castle"
        ]
        
        # Performance
        self.fps = 0
        self.fps_time = time.time()
        self.frame_count = 0
        
        # Ollama setup
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_available = self.check_ollama()
        
        # Animation
        self.particle_effects = []
        self.shake_offset = 0
        
        # Tutorial
        self.show_tutorial = True
        self.tutorial_step = 0

    def check_ollama(self):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                return any('phi3' in name.lower() for name in model_names)
            return False
        except:
            return False

    def is_index_finger_up(self, hand_landmarks):
        """Improved finger detection"""
        index_tip = hand_landmarks.landmark[8]
        index_pip = hand_landmarks.landmark[6]
        index_mcp = hand_landmarks.landmark[5]
        
        middle_tip = hand_landmarks.landmark[12]
        middle_pip = hand_landmarks.landmark[10]
        
        ring_tip = hand_landmarks.landmark[16]
        ring_pip = hand_landmarks.landmark[14]
        
        pinky_tip = hand_landmarks.landmark[20]
        pinky_pip = hand_landmarks.landmark[18]
        
        wrist = hand_landmarks.landmark[0]
        
        index_up = (index_tip.y < index_pip.y - 0.03) and (index_tip.y < index_mcp.y - 0.05)
        middle_down = middle_tip.y >= middle_pip.y - 0.02
        ring_down = ring_tip.y >= ring_pip.y - 0.02
        pinky_down = pinky_tip.y >= pinky_pip.y - 0.02
        index_extended = abs(index_tip.y - wrist.y) > 0.15
        
        return index_up and middle_down and ring_down and pinky_down and index_extended

    def get_smoothed_finger_pos(self, hand_landmarks, frame_shape):
        """Smoothed finger position"""
        h, w, _ = frame_shape
        index_tip = hand_landmarks.landmark[8]
        x = int(index_tip.x * w)
        y = int(index_tip.y * h)
        
        self.point_buffer.append((x, y))
        
        if len(self.point_buffer) >= 3:
            avg_x = int(sum(p[0] for p in self.point_buffer) / len(self.point_buffer))
            avg_y = int(sum(p[1] for p in self.point_buffer) / len(self.point_buffer))
            return (avg_x, avg_y)
        
        return (x, y)

    def analyze_drawing_advanced(self):
        """Advanced drawing analysis with pixel-based features"""
        if self.canvas is None or len(self.drawing_points) < 5:
            return None

        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Get all contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None

        # Main contour
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)
        
        if area < 500:
            return None
            
        perimeter = cv2.arcLength(main_contour, True)
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(main_contour)
        aspect_ratio = float(w) / h if h > 0 else 1
        
        # Convex hull
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Approximate polygon
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        num_vertices = len(approx)
        
        # Moments
        M = cv2.moments(main_contour)
        
        # Hu Moments for shape matching
        hu_moments = cv2.HuMoments(M).flatten()
        
        # Check for symmetry
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = w // 2, h // 2
        
        # Count separate components
        num_components = len(contours)
        
        # Check if closed shape
        is_closed = cv2.matchShapes(main_contour, hull, cv2.CONTOURS_MATCH_I1, 0) < 0.1
        
        # Detect specific shapes using template matching
        shape_type = self.detect_shape_type(circularity, num_vertices, aspect_ratio, solidity)
        
        # Complexity score
        complexity = len(self.drawing_points) / 1000.0
        
        # Fill ratio
        roi = thresh[y:y+h, x:x+w]
        if roi.size > 0:
            fill_ratio = np.sum(roi > 0) / roi.size
        else:
            fill_ratio = 0
        
        return {
            'shape_type': shape_type,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'circularity': circularity,
            'num_vertices': num_vertices,
            'area': area,
            'perimeter': perimeter,
            'num_components': num_components,
            'complexity': complexity,
            'fill_ratio': fill_ratio,
            'is_closed': is_closed,
            'width': w,
            'height': h,
            'centroid': (cx, cy)
        }

    def detect_shape_type(self, circularity, vertices, aspect_ratio, solidity):
        """Detect basic shape types"""
        if circularity > 0.85:
            return "circle"
        elif vertices == 3:
            return "triangle"
        elif vertices == 4:
            if 0.9 < aspect_ratio < 1.1:
                return "square"
            else:
                return "rectangle"
        elif vertices == 5:
            return "pentagon"
        elif vertices > 7 and solidity > 0.8:
            return "star"
        elif aspect_ratio > 2:
            return "horizontal_line"
        elif aspect_ratio < 0.5:
            return "vertical_line"
        else:
            return "irregular"

    def generate_multi_prompt(self, features):
        """Generate multiple AI prompts for better accuracy"""
        if not features:
            return []
        
        prompts = []
        
        # Prompt 1: Shape-based
        shape_desc = f"{features['shape_type']}"
        if features['num_components'] > 1:
            shape_desc += " with multiple parts"
        prompts.append(f"A child drew a {shape_desc}. Common objects with this shape:")
        
        # Prompt 2: Feature-based
        feature_desc = []
        if features['circularity'] > 0.7:
            feature_desc.append("round")
        if features['solidity'] > 0.9:
            feature_desc.append("solid filled")
        if features['aspect_ratio'] > 1.5:
            feature_desc.append("wide")
        elif features['aspect_ratio'] < 0.66:
            feature_desc.append("tall")
        
        if feature_desc:
            prompts.append(f"Drawing is {', '.join(feature_desc)}. What common object:")
        
        # Prompt 3: Size and complexity
        if features['complexity'] > 0.5:
            prompts.append(f"Detailed {features['shape_type']} drawing. Object:")
        else:
            prompts.append(f"Simple {features['shape_type']} shape. Object:")
        
        return prompts

    def call_ollama_multi_pass(self, features):
        """Multi-pass AI detection for better accuracy"""
        if not self.ollama_available:
            self.ai_guess = "⚠️ Ollama not running!"
            self.ai_processing = False
            return

        prompts = self.generate_multi_prompt(features)
        all_guesses = []
        
        # Try multiple prompts
        for prompt in prompts[:2]:  # Use first 2 prompts
            try:
                full_prompt = f"""{prompt}

Reply with ONLY 1-2 words - just the object name (like: sun, house, cat, tree, star, heart, car, face, flower).

Object is:"""

                payload = {
                    "model": "phi3:mini",
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 15,
                        "stop": ["\n", ".", "!", "Drawing"]
                    }
                }

                response = requests.post(self.ollama_url, json=payload, timeout=12)
                if response.status_code == 200:
                    result = response.json()
                    guess = result.get("response", "").strip()
                    guess = guess.split('\n')[0].strip().replace('"', '').replace("'", "")
                    
                    if len(guess) > 0 and len(guess) < 30:
                        all_guesses.append(guess.lower())
            except:
                continue
        
        # Find most common guess or use first
        if all_guesses:
            # Count occurrences
            guess_counts = {}
            for g in all_guesses:
                guess_counts[g] = guess_counts.get(g, 0) + 1
            
            # Get most common
            best_guess = max(guess_counts.items(), key=lambda x: x[1])[0]
            confidence = guess_counts[best_guess] / len(all_guesses)
            
            self.ai_confidence = confidence
            self.ai_guess = f"🎨 {best_guess.upper()}"
            
            # Add confidence indicator
            if confidence >= 0.8:
                self.ai_guess += " ✨"
            elif confidence >= 0.5:
                self.ai_guess += " 👍"
        else:
            self.ai_guess = "🤔 Try again!"
            self.ai_confidence = 0

        self.ai_processing = False

    def process_drawing_with_ai(self):
        """Process drawing with improved AI"""
        if self.ai_processing:
            return

        features = self.analyze_drawing_advanced()
        if not features:
            self.ai_guess = "⚠️ Draw something first!"
            return

        self.ai_processing = True
        self.ai_guess = "🤖 AI analyzing..."
        
        # Generate hint based on features
        self.hint_text = f"Hint: {features['shape_type']} shape"

        thread = threading.Thread(target=self.call_ollama_multi_pass, args=(features,))
        thread.daemon = True
        thread.start()

    def check_achievement(self, achievement_id):
        """Check and unlock achievements"""
        if not self.achievements[achievement_id]:
            self.achievements[achievement_id] = True
            self.new_achievement = achievement_id
            self.achievement_time = time.time()
            self.add_particle_effect("achievement")
            return True
        return False

    def add_particle_effect(self, effect_type):
        """Add visual particle effects"""
        if effect_type == "correct":
            for _ in range(15):
                self.particle_effects.append({
                    'x': random.randint(100, 1180),
                    'y': random.randint(100, 620),
                    'vx': random.uniform(-3, 3),
                    'vy': random.uniform(-5, -1),
                    'life': 1.0,
                    'color': random.choice([(0, 255, 0), (255, 255, 0), (0, 255, 255)])
                })
        elif effect_type == "achievement":
            for _ in range(30):
                self.particle_effects.append({
                    'x': 640,
                    'y': 360,
                    'vx': random.uniform(-5, 5),
                    'vy': random.uniform(-5, 5),
                    'life': 1.5,
                    'color': (255, 215, 0)
                })

    def update_particles(self):
        """Update particle animations"""
        for particle in self.particle_effects[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['vy'] += 0.2  # Gravity
            particle['life'] -= 0.02
            
            if particle['life'] <= 0:
                self.particle_effects.remove(particle)

    def draw_particles(self, frame):
        """Draw particle effects"""
        for particle in self.particle_effects:
            alpha = int(particle['life'] * 255)
            if alpha > 0:
                cv2.circle(frame, (int(particle['x']), int(particle['y'])), 
                          5, particle['color'], -1)

    def start_challenge_mode(self, difficulty="easy"):
        """Start challenge mode"""
        self.game_mode = "CHALLENGE"
        
        if difficulty == "easy":
            self.challenge_object = random.choice(self.easy_objects)
        elif difficulty == "medium":
            self.challenge_object = random.choice(self.medium_objects)
        else:
            self.challenge_object = random.choice(self.hard_objects)
        
        self.time_remaining = 60  # 60 seconds
        self.time_attack_start = time.time()

    def start_time_attack(self):
        """Start time attack mode"""
        self.game_mode = "TIME_ATTACK"
        self.time_remaining = 120  # 2 minutes
        self.time_attack_start = time.time()
        self.challenge_object = random.choice(self.easy_objects + self.medium_objects)

    def draw_modern_ui(self, frame):
        """Enhanced UI with game modes"""
        h, w, _ = frame.shape
        
        # Update FPS
        self.frame_count += 1
        if time.time() - self.fps_time > 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_time = time.time()

        # Top bar
        overlay = frame.copy()
        for i in range(100):
            alpha = 0.9 - (i / 100) * 0.4
            color_val = int(50 + (i / 100) * 30)
            cv2.line(overlay, (0, i), (w, i), (color_val, color_val, color_val), 1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Title
        title = "AI DRAWING GAME 🎨"
        if self.game_mode == "CHALLENGE":
            title = f"CHALLENGE: Draw a {self.challenge_object.upper()}!"
        elif self.game_mode == "TIME_ATTACK":
            title = f"TIME ATTACK: {int(self.time_remaining)}s left!"
        
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)[0]
        title_x = (w - title_size[0]) // 2
        
        # Glow effect
        for offset in range(2, 0, -1):
            cv2.putText(frame, title, (title_x, 45),
                       cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 100, 255), offset * 2)
        cv2.putText(frame, title, (title_x, 45),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)

        # Score panel
        score_text = f"SCORE: {self.score}"
        streak_text = f"STREAK: {self.streak}"
        
        cv2.rectangle(frame, (20, 10), (250, 80), (0, 100, 0), -1)
        cv2.rectangle(frame, (20, 10), (250, 80), (0, 255, 0), 3)
        cv2.putText(frame, score_text, (35, 38),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, streak_text, (35, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Drawing status
        status_color = (0, 255, 0) if self.is_drawing_mode else (100, 100, 100)
        cv2.circle(frame, (w - 40, 40), 15, status_color, -1)
        cv2.circle(frame, (w - 40, 40), 15, (255, 255, 255), 2)

        # AI Guess box
        if self.ai_guess:
            guess_y = 110
            
            words = self.ai_guess.split()
            lines = []
            current_line = []
            
            for word in words:
                test_line = " ".join(current_line + [word])
                text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)[0]
                if text_size[0] < w - 100:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(" ".join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(" ".join(current_line))
            
            box_height = len(lines) * 50 + 40
            
            # Colorful box
            if "analyzing" in self.ai_guess.lower():
                box_color = (150, 100, 0)
                border_color = (255, 200, 0)
            elif "⚠️" in self.ai_guess:
                box_color = (100, 50, 0)
                border_color = (200, 100, 0)
            else:
                box_color = (0, 120, 0)
                border_color = (0, 255, 0)
            
            cv2.rectangle(frame, (40, guess_y - 10), (w - 40, guess_y + box_height), 
                         box_color, -1)
            cv2.rectangle(frame, (40, guess_y - 10), (w - 40, guess_y + box_height), 
                         border_color, 4)
            
            for i, line in enumerate(lines):
                y_pos = guess_y + 30 + i * 50
                cv2.putText(frame, line, (60, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
            
            # Show hint if available
            if self.hint_text and self.ai_processing:
                hint_y = guess_y + box_height + 20
                cv2.putText(frame, self.hint_text, (60, hint_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Achievement notification
        if self.new_achievement and time.time() - self.achievement_time < 3:
            ach_names = {
                'first_guess': "🏆 First Guess!",
                'streak_3': "🔥 3 Streak!",
                'streak_5': "🔥🔥 5 Streak!",
                'streak_10': "🔥🔥🔥 10 STREAK!",
                'speed_demon': "⚡ Speed Demon!",
                'artist': "🎨 Artist!",
                'master': "👑 Master!"
            }
            ach_text = ach_names.get(self.new_achievement, "Achievement!")
            ach_y = h // 2 - 50
            
            cv2.rectangle(frame, (w//2 - 200, ach_y - 20), (w//2 + 200, ach_y + 40),
                         (100, 50, 150), -1)
            cv2.rectangle(frame, (w//2 - 200, ach_y - 20), (w//2 + 200, ach_y + 40),
                         (200, 100, 255), 4)
            
            text_size = cv2.getTextSize(ach_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
            text_x = w//2 - text_size[0]//2
            cv2.putText(frame, ach_text, (text_x, ach_y + 10),
                       cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

        # Bottom control panel
        panel_y = h - 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y), (w, h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Controls
        controls = [
            ("☝️ INDEX = Draw", (255, 255, 255)),
            ("✊ FIST = Stop", (200, 200, 200)),
        ]
        
        y_offset = panel_y + 25
        for text, color in controls:
            cv2.putText(frame, text, (30, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30

        # Keyboard shortcuts - more compact
        shortcuts = [
            ("[G] Guess", (0, 255, 255)),
            ("[Y] ✓", (0, 255, 0)),
            ("[C] Clear", (255, 200, 0)),
            ("[H] Hint", (200, 100, 255)),
            ("[1] Easy", (100, 255, 100)),
            ("[2] Med", (255, 255, 100)),
            ("[3] Hard", (255, 100, 100)),
            ("[T] Time", (100, 200, 255)),
            ("[Q] Quit", (255, 100, 100))
        ]
        
        x_pos = w - 850
        y_pos = h - 70
        for text, color in shortcuts:
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.putText(frame, text, (x_pos, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            x_pos += text_size[0] + 15
            if x_pos > w - 50:
                x_pos = w - 850
                y_pos += 25

        # Stats
        if self.total_drawings > 0:
            accuracy = int((self.correct_guesses / self.total_drawings) * 100)
            stats_text = f"Best Streak: {self.best_streak} | Accuracy: {accuracy}%"
            cv2.putText(frame, stats_text, (30, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Color indicator
        cv2.rectangle(frame, (w - 80, panel_y + 20), (w - 40, panel_y + 60),
                     self.draw_color, -1)
        cv2.rectangle(frame, (w - 80, panel_y + 20), (w - 40, panel_y + 60),
                     (255, 255, 255), 2)

    def handle_correct_answer(self):
        """Handle correct answer with animations and achievements"""
        if self.ai_guess and not self.ai_processing and "⚠️" not in self.ai_guess:
            self.score += 1
            self.streak += 1
            self.correct_guesses += 1
            self.total_drawings += 1
            
            if self.streak > self.best_streak:
                self.best_streak = self.streak
            
            # Add particles
            self.add_particle_effect("correct")
            
            # Check achievements
            if self.correct_guesses == 1:
                self.check_achievement('first_guess')
            if self.streak == 3:
                self.check_achievement('streak_3')
            if self.streak == 5:
                self.check_achievement('streak_5')
            if self.streak == 10:
                self.check_achievement('streak_10')
            
            # Clear for next
            self.canvas = np.zeros_like(self.canvas)
            self.drawing_points.clear()
            self.prev_point = None
            self.ai_guess = ""
            self.hint_text = ""
            
            # New challenge if in challenge mode
            if self.game_mode == "CHALLENGE":
                self.challenge_object = random.choice(self.easy_objects + self.medium_objects)
            elif self.game_mode == "TIME_ATTACK":
                self.challenge_object = random.choice(self.easy_objects + self.medium_objects)

    def run(self):
        """Main game loop"""
        print("=" * 60)
        print("AI DRAWING GAME V3 - SUPER ENHANCED")
        print("=" * 60)
        print("🎮 CONTROLS:")
        print("  ☝️  Index finger = Draw")
        print("  ✊ Fist = Stop")
        print("\n📱 GAME MODES:")
        print("  [1] Easy Challenge")
        print("  [2] Medium Challenge") 
        print("  [3] Hard Challenge")
        print("  [T] Time Attack")
        print("\n⌨️  KEYS:")
        print("  [G] AI Guess")
        print("  [Y] Correct (+1 score)")
        print("  [C] Clear canvas")
        print("  [H] Show hint")
        print("  [N] Change color")
        print("  [+/-] Brush size")
        print("  [F] Fullscreen")
        print("  [Q] Quit")
        print("=" * 60)

        if not self.ollama_available:
            print("\n⚠️  Ollama not running! Start: ollama serve")
        else:
            print("✅ Ollama ready\n")

        window_name = "AI Drawing Game V3"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            if self.canvas is None:
                self.canvas = np.zeros_like(frame)

            # Update time for timed modes
            if self.game_mode in ["CHALLENGE", "TIME_ATTACK"] and self.time_remaining > 0:
                elapsed = time.time() - self.time_attack_start
                self.time_remaining = max(0, 60 - int(elapsed)) if self.game_mode == "CHALLENGE" else max(0, 120 - int(elapsed))
                
                if self.time_remaining == 0:
                    print(f"Time's up! Score: {self.score}")
                    self.game_mode = "FREE"

            # Hand detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            self.is_drawing_mode = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2)
                    )

                    if self.is_index_finger_up(hand_landmarks):
                        self.is_drawing_mode = True
                        current_point = self.get_smoothed_finger_pos(hand_landmarks, frame.shape)

                        # Cursor with color
                        cv2.circle(frame, current_point, 15, self.draw_color, 3)
                        cv2.circle(frame, current_point, 10, (255, 255, 255), -1)

                        if self.prev_point is not None:
                            cv2.line(self.canvas, self.prev_point, current_point, 
                                   self.draw_color, self.draw_thickness, cv2.LINE_AA)
                            self.drawing_points.append(current_point)

                        self.prev_point = current_point
                    else:
                        self.prev_point = None
                        self.point_buffer.clear()
            else:
                self.prev_point = None
                self.point_buffer.clear()

            # Blend canvas
            frame_with_drawing = cv2.addWeighted(frame, 0.6, self.canvas, 0.4, 0)
            
            # Update and draw particles
            self.update_particles()
            self.draw_particles(frame_with_drawing)
            
            # Draw UI
            self.draw_modern_ui(frame_with_drawing)

            cv2.imshow(window_name, frame_with_drawing)

            # Keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print(f"\n👋 Game Over! Final Score: {self.score}")
                print(f"   Best Streak: {self.best_streak}")
                if self.total_drawings > 0:
                    print(f"   Accuracy: {int((self.correct_guesses/self.total_drawings)*100)}%")
                break
            elif key == ord('c'):
                self.canvas = np.zeros_like(frame)
                self.drawing_points.clear()
                self.prev_point = None
                self.ai_guess = ""
                self.hint_text = ""
            elif key == ord('g'):
                if len(self.drawing_points) > 10:
                    self.process_drawing_with_ai()
                else:
                    self.ai_guess = "⚠️ Draw something!"
            elif key == ord('y'):
                self.handle_correct_answer()
            elif key == ord('h'):
                features = self.analyze_drawing_advanced()
                if features:
                    self.hint_text = f"💡 It's a {features['shape_type']} shape!"
            elif key == ord('n'):
                self.current_color_idx = (self.current_color_idx + 1) % len(self.draw_colors)
                self.draw_color = self.draw_colors[self.current_color_idx]
            elif key == ord('=') or key == ord('+'):
                self.draw_thickness = min(20, self.draw_thickness + 2)
            elif key == ord('-') or key == ord('_'):
                self.draw_thickness = max(4, self.draw_thickness - 2)
            elif key == ord('1'):
                self.start_challenge_mode("easy")
            elif key == ord('2'):
                self.start_challenge_mode("medium")
            elif key == ord('3'):
                self.start_challenge_mode("hard")
            elif key == ord('t'):
                self.start_time_attack()
            elif key == ord('f'):
                current = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                if current == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = AIDrawingGame()
    game.run()