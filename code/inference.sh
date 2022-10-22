# python run/inference_no_filter.py  --breakpoint=1000 --sequence=1009_2 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 0 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=0930_1 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 0 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=0930_2 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 0 --conf-thres=0.4


# python run/inference_i_f.py --breakpoint=-1 --sequence=1009_7 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 0 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1009_4 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 0 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1009_5 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 0 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1009_4 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 0 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1009_6 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 0 --conf-thres=0.4

# python run/inference_i_f.py --breakpoint=-1 --sequence=0930_6 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 1 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=0930_5 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 1 --conf-thres=0.4

# python run/inference_i_f.py --breakpoint=-1 --sequence=1009_15 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 0 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1014_1 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 5 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1014_2 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 0 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1014_16 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 0 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1014_17 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 0 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1014_21 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 0 --conf-thres=0.4

for i in `seq 22`
do
# if [ $i -gt 2 ];then 
#     echo $i
# fi
python run/inference_i_f.py --breakpoint=-1 --sequence=1021_$i --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 0 --conf-thres=0.4
done