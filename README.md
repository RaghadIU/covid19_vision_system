# COVID-19 Vision Monitoring System

Introducing the COVID-19 Vision Monitoring System, developed as part of the Computer Vision course at IU International University of Applied Sciences.

---

## What is COVID-19 Vision Monitoring System? 
This tool help with monitor compliance with COVID-19 protocols in indoor public spaces. It allows users to:
- Detect humans in a video stream.
- Detect if individuals are wearing a mask.
- Estimate social distancing between people.
- Evaluate risk levels (Safe / High Risk) based on mask usage and proximity.
- Save annotated videos showing mask compliance and social distance violations.

video output make monitoring COVID-19 protocols simple and accessible.

---

## How to Install It?
1) Clone the Repository
```bash
git clone https://github.com/RaghadIU/covid19_vision_system.git
```
```bash
cd covid19_vision_system
```
2) Create and activate a virtual environment
```bash
python -m venv venv
```
```bash
venv\Scripts\activate
```
3) Install Python Dependencies 
Make sure you have Python 3.7+ installed. Then, run:
```bash
pip install -r requirements.txt
```

![Mask Detection](images/1.png)
(https://www.pexels.com/video/people-walking-on-sidewalk-5330835/)

## Mask detection , run:   
```bash
python -m mask_detection.mask_detector --source "videos/sample1.mp4" --out "outputs/mask_out.mp4" --view
```

![ Social distance estimator](images/2.png)
(https://www.pexels.com/video/black-and-white-video-of-people-853889/)

## Social distance estimator, run:   
```bash
python social_distance/distance_estimator.py --source "videos/sample2.mp4" --out "outputs/distance_out.mp4" --distance_factor 1.5 --view
```


![Human detection ](images/3.png)
(https://www.istockphoto.com/video/a-lot-people-wear-hygienic-mask-for-protect-pm2-5-dust-and-corona-virus-gm1207104496-348394435)
## Human detection , run:   
```bash
python human_detection/human_detector.py --source "videos/sample3.mp4" --out "outputs/human_out.mp4" --view
```



## few people 
![few people ](images/4.png)
(https://www.istockphoto.com/video/senior-couple-enjoying-taking-a-walk-in-green-city-gm2223072521-639304795)
## medium people 
![medium people ](images/5.png)
(https://www.istockphoto.com/video/grandparents-and-grandchildren-walking-during-covid-19-pandemic-gm1286451135-382951674)
## many people 
![many people ](images/6.png)
(https://www.istockphoto.com/video/crowd-of-people-commuting-to-work-gm1410259825-460481415)

## multi_model_mask_distance , run:   
```bash
python multi_model_mask_distance.py

```

