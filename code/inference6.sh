# python run/inference_i_f.py --breakpoint=-1 --sequence=1009_6 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 6 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1014_5 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 6 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1014_20 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 6 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1014_19 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 6 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1014_18 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 6 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1014_17 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 6 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1009_10 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 6 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1009_11 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 6 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1009_12 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 6 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=1014_21 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 6 --conf-thres=0.4
for i in `seq 151`
do
if [ $i -gt 132 ];then 
#     echo $i
# fi
    python run/inference_i_f.py --breakpoint=-1 --sequence=1021_$i --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 6 --conf-thres=0.4
fi
done