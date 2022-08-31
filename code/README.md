# 2D-detection For AIPE Basketball

> Given a set of  images or a video, detect the position of basketballs appearing in the scene.
>
> Based on a model trained with YOLOv5.



## Installation

Just install the packages in {requirements.txt}.



## Data Preparation

The data should be kept in a folder, placed at your project directory

See this is quite chill, you only need to make sure that the files you need to run are in one folder.



## Run Code

Just simply run the detect.py

`python detect.py --source {path/to/files/to_be_detected} --conf-thres {set threshold}`

To modified your own run, you shall modify few arguments:

| arg_name     | value                                 | description                                                  |
| ------------ | ------------------------------------- | ------------------------------------------------------------ |
| --weights    | path/to/your/own/model                | You may use your own model, but the default model is good enough probably ^_^. |
| --source     | path/to/your/own/files                | No need to say nothing.                                      |
| --output     | to/your/own/output/path               | No need to say nothing.                                      |
| --conf-thres | (0,1)                                 | Object confidence threshold. The default value is 0.5, you shall change the value to your own need. |
| --device     | cuda device, i.e. 0 or 0,1,2,3 or cpu | Devices used to run inference.                               |
| --iou-thres  | (0,1)                                 | The threshold of merging two boxes to one. The default value is 0.5, you shall change the value to your own need. |

Then, you should be able to see the code running in your terminal and showing detecting information. Done inferencing, the results are saved at the  `inference/output/{your/path}`.

You shall see two files in the folder discussed above:

1. A video showing the inference results with boxes on the basketball detected.

2. A text showing the position of the basketball in each frame(could be more than one basketball in one frame).

   1. The text is organized as follows:

      frame_no  xcenter  ycenter  width  height
