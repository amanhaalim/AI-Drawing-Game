#!/usr/bin/env python3
"""
System Check Script for AI Drawing Game V2
Checks all dependencies and configurations
"""

import sys
import subprocess

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_status(check, status, message=""):
    symbols = {"✓": "✓", "✗": "✗", "⚠": "⚠"}
    symbol = symbols.get(status, status)
    print(f"{symbol} {check}: {message}")

def check_python_version():
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_status("Python Version", "✓", f"{version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_status("Python Version", "✗", f"{version.major}.{version.minor}.{version.micro} - Need 3.8+")
        return False

def check_package(package_name):
    try:
        __import__(package_name)
        print_status(f"Package: {package_name}", "✓", "Installed")
        return True
    except ImportError:
        print_status(f"Package: {package_name}", "✗", "NOT installed")
        return False

def check_camera():
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print_status("Camera", "✓", f"Working ({frame.shape[1]}x{frame.shape[0]})")
                return True
            else:
                print_status("Camera", "✗", "Can't read frames")
                return False
        else:
            print_status("Camera", "✗", "Can't open camera")
            return False
    except Exception as e:
        print_status("Camera", "✗", str(e))
        return False

def check_ollama():
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = [m.get('name', '') for m in data.get('models', [])]
            
            if any('phi3' in name.lower() for name in models):
                print_status("Ollama Server", "✓", "Running with phi3:mini")
                return True
            else:
                print_status("Ollama Server", "⚠", "Running but phi3:mini not found")
                print("  → Run: ollama pull phi3:mini")
                return False
        else:
            print_status("Ollama Server", "✗", "Server error")
            return False
    except requests.exceptions.ConnectionError:
        print_status("Ollama Server", "✗", "Not running")
        print("  → Run: ollama serve")
        return False
    except Exception as e:
        print_status("Ollama Server", "✗", str(e))
        return False

def main():
    print_header("AI DRAWING GAME V2 - SYSTEM CHECK")
    
    all_checks = []
    
    # Python version
    print("\n📦 PYTHON CHECK")
    all_checks.append(check_python_version())
    
    # Required packages
    print("\n📚 PACKAGE CHECK")
    packages = ['cv2', 'mediapipe', 'numpy', 'requests']
    package_map = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'numpy': 'numpy',
        'requests': 'requests'
    }
    
    for pkg in packages:
        all_checks.append(check_package(pkg))
    
    # Camera
    print("\n📷 CAMERA CHECK")
    all_checks.append(check_camera())
    
    # Ollama
    print("\n🤖 AI SERVICE CHECK")
    all_checks.append(check_ollama())
    
    # Summary
    print_header("SUMMARY")
    passed = sum(all_checks)
    total = len(all_checks)
    
    print(f"\nChecks passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All checks passed! You're ready to play!")
        print("\nRun the game with:")
        print("  python ai_drawing_game_v2.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        
        if not all(all_checks[:4]):
            print("\n📦 To install missing packages:")
            print("  pip install opencv-python mediapipe numpy requests")
        
        if not all_checks[-1]:
            print("\n🤖 To setup Ollama:")
            print("  1. Download from: https://ollama.ai/download")
            print("  2. Start server: ollama serve")
            print("  3. Install model: ollama pull phi3:mini")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()