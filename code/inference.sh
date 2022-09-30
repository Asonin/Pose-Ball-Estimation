# python run/inference.py --sequence=team0817_1 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 0,1,2,3 --conf-thres=0.4
# python run/inference_euro_pose.py --sequence=team0817_1 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 8,9 --conf-thres=0.4
python run/inference_filtered.py --breakpoint=-1 --sequence=team0817_1 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 0,1,2,3 --conf-thres=0.4
# 人数变化切掉