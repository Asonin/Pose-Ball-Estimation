# python run/inference.py --sequence=0930_5 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 8,9 --conf-thres=0.4
# python run/inference_euro_pose.py --sequence=team0817_1 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 8,9 --conf-thres=0.4
python run/inference_framebyframe.py --breakpoint=-1 --sequence=0930_5 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 9 --conf-thres=0.4
python run/inference_framebyframe.py --breakpoint=-1 --sequence=0930_6 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 9 --conf-thres=0.4
python run/inference_framebyframe.py --breakpoint=-1 --sequence=0930_7 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 9 --conf-thres=0.4
python run/inference_framebyframe.py --breakpoint=-1 --sequence=0930_8 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 9 --conf-thres=0.4
python run/inference_framebyframe.py --breakpoint=-1 --sequence=0930_9 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 9 --conf-thres=0.4
