# 📷 Camera Setup Guide - Using USB Webcam with Zenify

This guide explains how to use your USB webcam instead of the built-in camera with Zenify.

## 🎯 Quick Start

### Option 1: Web Interface (Easiest)

1. **Start Zenify**:
   ```bash
   python app.py
   ```

2. **Open your browser**: Go to http://localhost:5000

3. **Select Camera**:
   - Scroll down to "📷 Camera Settings" section
   - Click **"Select Camera"** button
   - Choose your USB webcam from the list
   - Click **"Select"**

4. **Done!** Your selection is saved automatically.

---

### Option 2: Interactive Camera Tester (Recommended for Testing)

Run the enhanced camera detection tool:

```bash
python test_cameras.py
```

**Features**:
- 🔍 Automatically detects all cameras (built-in + USB)
- 📹 Test each camera with live preview
- 💾 Save your camera preference
- ℹ️ Shows resolution and backend info

**Usage**:
1. The script will scan and list all available cameras
2. Type a number (1, 2, etc.) to test that camera with live preview
3. Press 'q' to close the preview window
4. Type 's' to save your camera selection for Zenify
5. Type 'q' to quit

---

### Option 3: Command-Line Selection

```bash
python select_camera.py
```

This will:
- Detect all available cameras
- Show you a numbered list
- Let you select which one to use
- Save your choice to `camera_config.json`

---

### Option 4: Manual Configuration

Create a file named `camera_config.json` in the Zenify root directory:

```json
{
  "camera_index": 1,
  "backend": "CAP_MSMF"
}
```

**Camera Index Guide**:
- `0` = Built-in camera (usually)
- `1` = First USB webcam (usually)
- `2` = Second USB webcam (if you have multiple)

---

## 🔔 Automatic Camera Prompt

When you start a session, Zenify will now:
- ✅ Automatically detect all available cameras
- ✅ Prompt you if multiple cameras are found
- ✅ Let you choose which camera to use
- ✅ Remember your choice for future sessions

---

## 🧪 Testing Your Camera

### Test with the enhanced test script:

```bash
python test_emotion_improvements.py
```

This will:
1. Detect all available cameras
2. Ask you to select which camera to test
3. Run performance tests on the selected camera
4. Show FPS and emotion detection accuracy

---

## 🐛 Troubleshooting

### USB Webcam Not Detected?

1. **Check if it's plugged in**
   - Make sure the USB cable is securely connected
   - Try a different USB port (preferably USB 3.0)

2. **Close other applications**
   - Close Zoom, Teams, Skype, or any app using the camera
   - Close the Windows Camera app if it's open

3. **Run the camera detector**
   ```bash
   python test_cameras.py
   ```
   This will show you exactly which cameras are detected.

4. **Check Device Manager (Windows)**
   - Press `Win + X` → Device Manager
   - Look under "Cameras" or "Imaging devices"
   - Make sure your USB webcam is listed and has no warning icons

5. **Try unplugging and replugging**
   - Unplug the USB webcam
   - Wait 5 seconds
   - Plug it back in
   - Run `python test_cameras.py` again

6. **Update drivers**
   - Right-click on your webcam in Device Manager
   - Select "Update driver"
   - Choose "Search automatically for drivers"

---

## 📊 Camera Information

### How to check which camera is currently being used:

1. **In the web interface**:
   - Go to http://localhost:5000
   - Click "Select Camera"
   - The currently selected camera will be highlighted in green

2. **Check the config file**:
   - Open `camera_config.json` in the Zenify folder
   - Look at the `camera_index` value

3. **Check the console output**:
   - When you start `python app.py`, it will print:
     ```
     📷 Camera opened successfully: Index 1, Backend: CAP_MSMF
     ```

---

## 💡 Tips

- **USB 3.0 ports** provide better performance for high-resolution webcams
- **Close unnecessary apps** to free up camera resources
- **Good lighting** improves emotion detection accuracy
- **Position the camera** at eye level for best results
- **Test your camera** before starting an important session

---

## 🎥 Camera Comparison

| Feature | Built-in Camera | USB Webcam |
|---------|----------------|------------|
| Resolution | Usually 720p | Often 1080p or higher |
| Positioning | Fixed | Adjustable |
| Quality | Basic | Usually better |
| Portability | Built-in | Requires USB port |

---

## ❓ FAQ

**Q: Can I use multiple USB webcams?**  
A: Yes! Zenify can detect and use multiple cameras. Just select the one you want.

**Q: Will my camera selection be remembered?**  
A: Yes, your choice is saved in `camera_config.json` and will be used automatically next time.

**Q: Can I switch cameras during a session?**  
A: Currently, you need to finish the session and select a different camera before starting a new one.

**Q: What if I don't see my USB webcam in the list?**  
A: Run `python test_cameras.py` to diagnose the issue. Make sure no other app is using the camera.

**Q: Which camera index is my USB webcam?**  
A: Usually index 1, but run `python test_cameras.py` to see all detected cameras and their indices.

---

## 📞 Need Help?

If you're still having issues:
1. Run `python test_cameras.py` and share the output
2. Check if the camera works in other apps (Windows Camera, Zoom, etc.)
3. Make sure you have the latest version of OpenCV: `pip install --upgrade opencv-python`

---

**Happy stress monitoring with your USB webcam! 🎉**