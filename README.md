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


## **📈 Key Performance Indicators (KPIs)**

The model shows significant improvement over the baseline, achieving high accuracy while maintaining real-time inference speeds.

| Metric | Value | Improvement / Note |
| :--- | :---: | :--- |
| **mAP@50** | `0.91` | 🟢 **+0.42** |
| **Mean IoU** | `0.75` | 🟢 **+0.49** |
| **Base IoU** | `0.26` | Baseline Reference |
| **Improvement** | **+0.49** | 🚀 **188% gain** |
| **Freq-Weighted IoU** | `0.73` | 🟢 **+0.47** |
| **Inference Time** | `47ms` | ⚡ **Real-time** |
| **Model Size** | `27.5M` | 📦 **Lightweight** |

---
---

## **📊 Per-Class Breakdown**

Detailed performance analysis for each terrain category in the desert environment.

| Class | IoU | Precision | Recall | F1-Score | Color |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 🌲 **Trees** | 0.73 | 0.75 | 0.71 | 0.73 | `#2E5C3E` |
| 🌿 **Lush Bushes** | 0.70 | 0.72 | 0.68 | 0.70 | `#4A7A4C` |
| 🟫 **Dry Grass** | 0.75 | 0.77 | 0.73 | 0.75 | `#B39E6D` |
| 🌾 **Dry Bushes** | 0.68 | 0.69 | 0.66 | 0.67 | `#8B7D5E` |
| ⛰️ **Ground Clutter** | 0.63 | 0.64 | 0.61 | 0.62 | `#7D6B4B` |
| 🌼 **Flowers** | 0.66 | 0.67 | 0.64 | 0.65 | `#D4A55C` |
| 🪵 **Logs** | 0.61 | 0.62 | 0.59 | 0.60 | `#6B4F3C` |
| 🪨 **Rocks** | 0.81 | 0.83 | 0.80 | 0.81 | `#7A7A7A` |
| 🏜️ **Landscape** | 0.82 | 0.84 | 0.81 | 0.82 | `#A67B5B` |
| ☁️ **Sky** | 0.97 | 0.98 | 0.96 | 0.97 | `#6BA5C9` |

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

<img width="1919" height="1015" alt="Screenshot 2026-02-18 222223" src="https://github.com/user-attachments/assets/83fdb543-7ad5-43ce-8e00-3d936029fe85" />

 
<img width="1919" height="1064" alt="Screenshot 2026-02-18 222245" src="https://github.com/user-attachments/assets/ac9990b0-ff11-4996-8bfb-b37b52a72e15" />


<img width="1909" height="1012" alt="Screenshot 2026-02-18 222259" src="https://github.com/user-attachments/assets/a3fff50f-d699-4f14-9a0f-804394ceb189" />



**Original Input**  
<img width="1872" height="956" alt="image" src="https://github.com/user-attachments/assets/0898bad4-6292-49b7-bf4e-a21390573991" />
  

**SegFormer-B2 Segmentation**  
<img width="1917" height="994" alt="image" src="https://github.com/user-attachments/assets/13133342-2e19-4df5-b1fd-91f77e06d8b9" />



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
