# POSE_BALL_ESTIMATION

> Given a set of images or a video, detect the position of basketballs appearing in the scene.
>
> Based on a model trained with YOLOv5.



## Installation

Just install the packages in {requirements.txt} and probably a bit more of .



## Run Code

Just simply run the run/_inference.py_

e.g.

>  `python run/inference.py --sequence=your_prefix --scene=your_config --weights route/to/the/weights_file --extrinsics_path route/to/your/extrinsic_paths --device choose_available_devices`

To modified your own run, you shall modify few arguments:

| arg_name     | value                                 | description                                                  |
| ------------ | ------------------------------------- | ------------------------------------------------------------ |
| --weights    | path/to/your/own/model                | You may use your own model, but the default model is good enough probably ^_^. |
| --sequence     | path/to/your/own/files                | No need to say nothing.                                      |
| --output     | to/your/own/output/output_path               | No need to say nothing.                                      |
| --conf-thres | (0,1)                                 | Object confidence threshold. The default value is 0.5, you shall change the value to your own need. |
| --device     | cuda device, i.e. 0 or 0,1,2,3 or cpu | Devices used to run, currently only supporting single GPU inference.                               |
| --extrensics_path  | path/to/the/camera_extrinsics                                 | the camera extrinsics |
| --num_people | [1,10] | number of people showing up in the sequence, determines the detection results| 
Then, you should be able to see the code running in your terminal and showing detecting information. Done inferencing, the results are saved at the  `../output/{your/path}`.

You shall see two files in the folder discussed above:

1. A video showing the inference results with boxes on the basketball detected.

2. A text showing the position of the basketball in each frame(could be more than one basketball in one frame).

   1. The text is organized as follows:

      frame_no  xcenter  ycenter  width  height
