# PitchSense
A computer vision and machine learning project which tracks a pitch from a video, extracts ball trajectory features, and classifies the pitch into a fastball, curveball, or changeup. The script utilizes OpenCV, Python, and CUDA to aquire the desired result.

# Basic Command Line code to run .py files

## To run the whole program
```powershell
python .\pipeline.py --video "C:\Users\natei\Downloads\Capcut Folder\testpitch2.mov" --pitch-id "test001" --pitch-type "unknown" --pitcher-handedness "RHP" --roi 900 350 2300 1450 --start-frame 12 --end-frame 110 --manual-anchor-frames 12 18 24 30 36 42 48 54 60 69 78 86 95 105 110 --training-csv ".\compiled_pitch_features_pathshape_30samples.csv" --model ".\classifier_output_pathshape\best_pitch_classifier_pathshape.joblib"
```
## To run the CUDA version
```powershell
nvcc main.cu kernel.cu -o cuda_motion

./cuda_motion
```
## ball_tracker.py

Format: 
```powershell
python ball_tracker_model.py --video "path_to_video.mov" --save-dataset --pitch-id "pitch001" --pitch-type "fastball"
```

### Ex.
```powershell
python .\ball_tracker_model.py --video "C:\Users\natei\Downloads\Capcut Folder\pitch1(11).mov" --roi 900 350 2300 1450 --start-frame 20 --end-frame 110 --manual-anchor-frames 24 32 40 48 56 64 72 80 88 95 105 110 --pitch-direction right --corridor-radius 70 --search-radius 60 --display --debug --save-dataset --dataset-dir ".\pitch_dataset" --pitch-id "pitch001" --pitch-type "unknown" --output ".\tracked_pitch_v11.mp4" --csv ".\ball_path_v11.csv" --metrics ".\pitch_metrics_v11.txt"
```
## train_pitch_classifier.py
Format: 
```powershell
python train_pitch_classifier.py --features ".\compiled_pitch_features_30samples.csv" --out-dir ".\classifier_output"
```
### Ex.
```powershell
python .\train_pitch_classifier.py --features "C:\Users\natei\OneDrive\Documents\Desktop\GPITCHU\compiled_updated_pitch_features_30samples.csv" --out-dir ".\classifier_output"
```
## predict_pitch_type.py
Format: 
```powershell
python predict_pitch_type.py --model ".\classifier_output\best_pitch_classifier.joblib" --features ".\compiled_pitch_features_30samples.csv"
```
### Ex.
```powershell
python .\predict_pitch_type.py --model ".\classifier_output\best_pitch_classifier.joblib" --features ".\compiled_updated_pitch_features_30samples.csv" --output ".\pitch_predictions.csv"
```

# Additional Information
Each folder contains information pertaining to the process of creating, training, and speeding up the pitch classifier.
- classifier_output contains the trained model, accuracy of the model, training data cleaned, etc.
- cuda_acceleration contains the .cu files to speed up the classifier model
- pitch_metrics contains the dataset used to train the model
