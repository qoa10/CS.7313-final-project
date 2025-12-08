# CS.7313-final-project

## Dataset  

The full dataset (≈ 10 GB) is hosted on Google Drive — download it here:  
https://drive.google.com/drive/folders/1o0CEVT2s0Pl-W_71onLo3iuziuupjb1T?usp=drive_link  

After downloading, place the dataset folder anywhere on your computer. Then update the paths at the top of the scripts (e.g. `DATA_ROOT`) so they point to your local dataset root.  

---  

## Quick Setup (PyCharm + virtual environment)  

1. Clone this repo:  
   `git clone https://github.com/<your-username>/CS.7313-final-project.git`  
2. Open the folder in PyCharm.  
3. Create a new Python virtual environment (Python 3.9–3.11 recommended).  
4. Install dependencies (e.g. `torch`, `torchvision`, `opencv-python`, `pycocotools`, `ultralytics`, `numpy`, `pandas`, etc.).  
5. Edit the dataset path variables in the data-processing and training scripts so they refer to your local dataset folder.  

---  

With that done, you can run the scripts in order to reproduce data preparation and experiments.  
