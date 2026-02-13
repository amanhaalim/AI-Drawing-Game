# AI Drawing Guessing Game

A fun interactive game where they draw in the air with their finger and AI guesses what they drew!

## Requirements

- Windows 11
- Python 3.8 or higher
- Ollama installed locally
- Webcam

## Installation Steps

### Step 1: Install Python Dependencies

Open Command Prompt and navigate to the project folder, then run:

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install opencv-python==4.8.1.78
pip install mediapipe==0.10.8
pip install numpy==1.24.3
pip install requests==2.31.0
```

### Step 2: Start Ollama

Open a NEW Command Prompt window and run:

```bash
ollama serve
```

Keep this window open while playing the game.

If you haven't pulled the phi3:mini model yet, run:

```bash
ollama pull phi3:mini
```

### Step 3: Run the Game

In your original Command Prompt window:

```bash
python ai_drawing_game.py
```

## How to Play

1. **Draw in the air**: Point with your INDEX FINGER to draw
2. **Stop drawing**: Make a FIST to stop
3. **Get AI guess**: Press 'G' key to ask AI what you drew
4. **Mark correct**: Press 'Y' if AI guessed correctly (+1 to score)
5. **Clear canvas**: Press 'C' to clear and draw again
6. **Quit**: Press 'Q' to exit

## Controls Reference

- **Index Finger Up** = Drawing mode (green dot appears)
- **Fist** = Stop drawing
- **G Key** = Ask AI to guess your drawing
- **Y Key** = Confirm AI guess is correct (adds point)
- **C Key** = Clear canvas
- **Q Key** = Quit game

## Troubleshooting

### "Ollama not running!" warning

- Make sure Ollama is running in a separate terminal: `ollama serve`
- Check that Ollama API is accessible at: http://localhost:11434

### Camera not working

- Make sure no other application is using the webcam
- Try restarting the script

### Slow performance

- Close other applications to free up RAM
- The script is optimized for 8GB RAM

### AI responses are slow

- This is normal with phi3:mini on CPU
- The camera feed won't freeze (uses threading)

## Features

- Real-time hand tracking
- Air drawing with index finger
- AI-powered drawing recognition
- Score tracking
- Child-friendly interface
- Optimized for low-resource machines
- Non-blocking AI calls (camera stays smooth)

## Technical Details

- Uses MediaPipe for hand tracking
- OpenCV for video processing
- Ollama phi3:mini for AI guessing
- Threading to prevent camera freeze
- Optimized for 8GB RAM systems
