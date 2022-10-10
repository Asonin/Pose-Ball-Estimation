# python run/inference.py --sequence=0930_5 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 8,9 --conf-thres=0.4
# python run/inference_euro_pose.py --sequence=team0817_1 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 8,9 --conf-thres=0.4
# python run/inference_framebyframe.py --breakpoint=-1 --sequence=1009_1 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 8 --conf-thres=0.4
python run/inference_framebyframe.py --breakpoint=-1 --sequence=1009_2 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 8 --conf-thres=0.4
python run/inference_framebyframe.py --breakpoint=-1 --sequence=1009_3 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 8 --conf-thres=0.4
python run/inference_framebyframe.py --breakpoint=-1 --sequence=1009_4 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 8 --conf-thres=0.4
python run/inference_framebyframe.py --breakpoint=-1 --sequence=1009_5 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 8 --conf-thres=0.4
python run/inference_framebyframe.py --breakpoint=-1 --sequence=1009_6 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 8 --conf-thres=0.4
