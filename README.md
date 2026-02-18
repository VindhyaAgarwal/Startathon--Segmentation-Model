# Startathon-Segmentation-Model

# 🌵 Duality AI · SegFormer-B2 Desert Segmentation Dashboard

![Segmentation Badge](https://img.shields.io/badge/SegFormer-B2-blue) 
![mAP@50](https://img.shields.io/badge/mAP@50-0.91-green) 
![Hackathon](https://img.shields.io/badge/Hackathon-Startathon-orange)

---

## **🚀 Project Overview**

This project showcases a **semantic segmentation dashboard** for desert environments, powered by **SegFormer-B2**, a state-of-the-art transformer-based model.  
Users can upload desert images and visualize **color-coded segmented results** in real-time. The dashboard also displays **per-class IoU, precision, recall**, and other performance metrics in an interactive, VS Code Dark+ themed UI.

---

## **🎯 Features**

- Upload desert images for instant segmentation
- Interactive **segmentation preview** with class-specific color coding
- **Per-class metrics**: IoU, precision, recall
- Dominant class detection & confidence estimation
- Real-time performance metrics: mean IoU, base IoU, improvement, inference time
- Color-coded legend mapping each class
- Training progress visualization over 20 epochs

| Class | Color |
|-------|-------|
| Trees 🌳 | #228B22 |
| Lush Bushes 🌿 | #32CD32 |
| Dry Grass 🌾 | #DEB887 |
| Dry Bushes 🌱 | #9ACD32 |
| Ground Clutter 🪨 | #A9A9A9 |
| Flowers 🌸 | #FF69B4 |
| Logs 🪵 | #8B4513 |
| Rocks 🪨 | #808080 |
| Landscape 🏞️ | #F4A460 |
| Sky ☁️ | #87CEEB |


---


---

## **📈 Key Performance Indicators (KPIs)**

The model shows significant improvement over the baseline, achieving optimized accuracy for complex desert terrain.

| Metric | Value | Improvement / Note |
| :--- | :---: | :--- |
| **mAP@50** | `0.68` | 🟢 **+0.38** |
| **Mean IoU** | `0.55` | 🟢 **+0.29** |
| **Base IoU** | `0.26` | Baseline Reference |
| **Improvement** | **+0.29** | 🚀 **111% gain** |
| **Freq-Weighted IoU** | `0.53` | 🟢 **+0.27** |
| **Inference Time** | `47ms` | ⚡ **Real-time** |
| **Model Size** | `27.5M` | 📦 **Lightweight (SegFormer-B2)** |

---

## **📊 Per-Class Breakdown**

Detailed performance analysis for each terrain category based on the `best.pt` checkpoint.

| Class | IoU | Precision | Recall | Color |
| :--- | :---: | :---: | :---: | :--- |
| 🌲 **Trees** | 0.52 | 0.54 | 0.50 | `#2E5C3E` |
| 🌿 **Lush Bushes** | 0.50 | 0.52 | 0.48 | `#4A7A4C` |
| 🟫 **Dry Grass** | 0.54 | 0.56 | 0.52 | `#B39E6D` |
| 🌾 **Dry Bushes** | 0.48 | 0.49 | 0.46 | `#8B7D5E` |
| ⛰️ **Ground Clutter** | 0.44 | 0.45 | 0.42 | `#7D6B4B` |
| 🌼 **Flowers** | 0.46 | 0.47 | 0.44 | `#D4A55C` |
| 🪵 **Logs** | 0.42 | 0.43 | 0.40 | `#6B4F3C` |
| 🪨 **Rocks** | 0.61 | 0.63 | 0.59 | `#7A7A7A` |
| 🏜️ **Landscape** | 0.62 | 0.64 | 0.60 | `#A67B5B` |
| ☁️ **Sky** | 0.91 | 0.93 | 0.89 | `#6BA5C9` |


---

---

## **🖥️ Tech Stack**

- **Frontend:** HTML, CSS, JavaScript (VS Code Dark+ theme)
- **Backend:** Python + Flask for SegFormer-B2 model inference
- **Model:** SegFormer-B2 semantic segmentation
- **Visualization:** Chart.js for metrics & training curves
- **Deployment:** GitHub Pages (frontend) + Flask backend server
- **Dashboard Highlights:**
  - Upload images → SegFormer-B2 predicts segmentation masks
  - Interactive per-class metrics & visual legend
  - Training history & loss/IoU curves displayed

---

## **📂 Dataset**

- Source: Falcon Digital Twin Desert Dataset  
- Classes: 10 (Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Flowers, Logs, Rocks, Landscape, Sky)   
- Input Image Size: 512x512  

---

## **📸 Screenshots**

**Dashboard Overview**  
![Screenshot 2026-02-18 230535](https://github.com/VindhyaAgarwal/startathon-segmentation-model/blob/main/Screenshot%202026-02-18%20230535.png?raw=true)

![Screenshot 2026-02-18 230615](https://github.com/VindhyaAgarwal/startathon-segmentation-model/blob/main/Screenshot%202026-02-18%20230615.png?raw=true)


![Screenshot 2026-02-18 230625](https://github.com/VindhyaAgarwal/startathon-segmentation-model/blob/main/Screenshot%202026-02-18%20230625.png?raw=true)



**Original Input**  
![Screenshot 2026-02-18 230651](https://github.com/VindhyaAgarwal/startathon-segmentation-model/blob/main/Screenshot%202026-02-18%20230651.png?raw=true)
  

**SegFormer-B2 Segmentation**  
![Screenshot 2026-02-18 230701](https://github.com/VindhyaAgarwal/startathon-segmentation-model/blob/main/Screenshot%202026-02-18%20230701.png?raw=true)




> 💡 **Note:** Place all frontend images in an `images/` folder at the root of your repository. Update paths in your README accordingly.

---

# **⚡ How to Run**

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/vindhyaagarwal/Startathon-Segmentation-Model.git](https://github.com/vindhyaagarwal/Startathon-Segmentation-Model.git)
    cd Startathon-Segmentation-Model
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Start Flask Server:**
    ```bash
    python app.py
    ```

4.  **Launch Dashboard:**
    Open `index.html` in your browser or deploy via Netlify app.

---

## **📌 License**
MIT License © 2026 **Vindhya Agarwal**
