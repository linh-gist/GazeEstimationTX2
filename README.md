> [**Optimal Camera Position for a Practical Application of Gaze Estimation on Edge Devices**](),            
> Linh Van Ma, Tin Trung Tran, Moongu Jeon

## How to run?
If you want to finetune this deep learning model. You first need to collect your dataset.
You need to look at the center of each rectangle (36 rectangles). 

    python3 collect_dataset.py
    
Once you finish collecting your dataset. You need to change the folder of `subject` in `run_finetune.py`.
Then, you can start finetuning this deep learning model.

    python3 run_finetune.py

Remember to rebuild TensorRT if you first run this source in your device. You need to move your working folder to `ext\tensorrt_mtcnn`.
    
    chmod +x ./build.sh
    ./build.sh
    
You now can run to test this gaze estimation by first connect a realsense camera to Jetson TX2.
Run the following script.

    python3 run_camera.py

To test with your recorded video, you should specify you video location in `run_camera_test.py`.
Run the following script.

    python3 run_camera_test.py


## Dependencies
0. **FAZE: Few-Shot Adaptive Gaze Estimation**: https://github.com/NVlabs/few_shot_gaze
1. **eos**: https://github.com/patrikhuber/eos
2. **HRNets**: https://github.com/HRNet/HRNet-Facial-Landmark-Detection
3. **mtcnn-pytorch**: https://github.com/TropComplique/mtcnn-pytorch
4. **Realtime-facial-landmark-detection**: https://github.com/pathak-ashutosh/Realtime-facial-landmark-detection
5. **MTCNN TensorRT(Demo #2: MTCNN)**: https://github.com/jkjung-avt/tensorrt_demos#mtcnn
    
    5.1 [TensorRT MTCNN Face Detector](https://jkjung-avt.github.io/tensorrt-mtcnn/)
    
    5.2 [Optimizing TensorRT MTCNN](https://jkjung-avt.github.io/optimize-mtcnn/)

## Acknowledgement
A large part of the code is borrowed from [FAZE: Few-Shot Adaptive Gaze Estimation](https://github.com/NVlabs/few_shot_gaze) and [MTCNN TensorRT(Demo #2: MTCNN)](https://github.com/jkjung-avt/tensorrt_demos#mtcnn). Thanks for their wonderful works.
