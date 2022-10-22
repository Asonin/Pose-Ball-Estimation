# python run/inference_i_f.py --breakpoint=-1 --sequence=0930_9 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 4 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1009_1 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 4 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1009_8 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 4 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1009_9 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 4 --conf-thres=0.4

# given to 6
# python run/inference_i_f.py --breakpoint=-1 --sequence=1009_7 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 4 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1014_7 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 4 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1014_8 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 4 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1014_9 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 4 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1014_10 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 4 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1014_11 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 4 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1014_8 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 4 --conf-thres=0.4
for i in `seq 110`
do
if [ $i -gt 88 ];then 
#     echo $i
# fi
    python run/inference_i_f.py --breakpoint=-1 --sequence=1021_$i --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 4 --conf-thres=0.4
fi
done