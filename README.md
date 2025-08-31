# COVID-19 Health Protocol Monitoring System  

## 📖 Project Overview
This project implements a **computer vision system** to monitor COVID-19 health protocols in indoor public spaces.  
The system can:  
- Detect humans in video sequences.  
- Estimate approximate distances between people to check **social distancing**.  
- Detect whether individuals are wearing **face masks**.  
- Identify violations of COVID-19 protocols (e.g., no mask + too close).  

---

## 🛠️ Technologies Used
- **Python 3.10+**
- [OpenCV](https://opencv.org/) (image processing & video handling)  
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (SoA human detection)  
- Pretrained **Mask Detection Model** (fine-tuned on public datasets)  
- NumPy, Matplotlib (data processing & visualization)

---

## 📂 Project Structure
