# 🌟 Baby Steps: Guide to Andromeda Star Classification 🌟

Hello! Welcome to our space-telescope adventure! 🚀 

This project uses artificial intelligence to look at **variable stars** (stars that change their brightness, like blinking lights in the night sky) and guess what category they belong to. 

Here is the exact order to read the files, explained as if you are starting from absolute scratch!

---

## 🗺️ The Tour Map (Order to Read the Files)

### 1. 📥 [download_light_curves.py](file:///Users/kima/CodeProjects/VariableStarsDetectionAlgorithm/download_light_curves.py)
* **What is it?** The "Space Shopper".
* **Simple Explanation:** It makes a phone call to the MAST database (a huge digital library of space telescope photos and measurements) and asks for raw observations near the Andromeda Galaxy. It downloads them as FITS files (which are just special tables astronomers use instead of standard Excel files).

### 2. 🧱 [src/model.py](file:///Users/kima/CodeProjects/VariableStarsDetectionAlgorithm/src/model.py)
* **What is it?** The "Robot Brain".
* **Simple Explanation:** It defines our CNN (Convolutional Neural Network) model. We stack Lego bricks like `Conv1d` (to scan for patterns in brightness), `BatchNorm` (to wash the numbers clean), `ReLU` (to silence sad negative numbers), and `Dropout` (to play hide-and-seek so the brain learns robustly).

### 3. 🧼 [src/data_setup.py](file:///Users/kima/CodeProjects/VariableStarsDetectionAlgorithm/src/data_setup.py)
* **What is it?** The "Data Laundry".
* **Simple Explanation:** Raw telescope data is noisy and messy! Some observations happen days apart, others weeks apart. This script:
  1. Finds the star's blinking rhythm (frequency) using a period finder.
  2. Wraps the timeline into a single repeating circle (phase folding).
  3. Stretches it out into exactly 100 neat points so the model can inspect it easily.

### 4. 🎓 [src/engine.py](file:///Users/kima/CodeProjects/VariableStarsDetectionAlgorithm/src/engine.py)
* **What is it?** The "Teacher".
* **Simple Explanation:** It teaches the model. It feeds the model data, listens to its guesses, checks the right answers, calculates how wrong the guesses were (loss), and adjusts the model's synapses so it does better next time.

### 5. 🏫 [train.py](file:///Users/kima/CodeProjects/VariableStarsDetectionAlgorithm/train.py)
* **What is it?** The "School Term".
* **Simple Explanation:** It runs the training loop 30 times (epochs) using a simulated school of 5000 stars, then saves the graduated model's weights to `models/star_classifier.pth`.

### 6. 🔍 [predict.py](file:///Users/kima/CodeProjects/VariableStarsDetectionAlgorithm/predict.py)
* **What is it?** The "Stethoscope Test".
* **Simple Explanation:** Want to check a single star file? Pass it here, and it will draw a pretty 3-panel picture (`lightcurve_diagnostic_plot.png`) showing the raw observations, the period peak, and the folded curve, along with the model's guess.

### 7. 🗂️ [classify_unlabeled.py](file:///Users/kima/CodeProjects/VariableStarsDetectionAlgorithm/classify_unlabeled.py)
* **What is it?** The "Factory Cataloger".
* **Simple Explanation:** It takes all the raw FITS files downloaded by the Space Shopper, cleans them up, feeds them to the trained Robot Brain, compiles all predictions into a neat spreadsheet (`data/classified_stars_catalog.csv`), and saves diagnostic charts for each.

### 8. 🎮 [app.py](file:///Users/kima/CodeProjects/VariableStarsDetectionAlgorithm/app.py)
* **What is it?** The "Game Console".
* **Simple Explanation:** It runs a web app in your browser using Gradio. You can drag-and-drop a star file, and it instantly prints the model's prediction with a neat chart!

---

## 🚀 How to Run the App (Super Easy!)

Open your terminal and run these steps:

1. **Get the Data:**
   ```bash
   python3 download_light_curves.py
   ```
2. **Train the Brain:**
   ```bash
   python3 train.py
   ```
3. **Classify Real Stars:**
   ```bash
   python3 classify_unlabeled.py
   ```
4. **Play with the Web App:**
   ```bash
   python3 app.py
   ```
   *Then open the link (usually http://127.0.0.1:7860) printed on your screen!*
