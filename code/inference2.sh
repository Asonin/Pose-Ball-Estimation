# python run/inference.py --sequence=0930_5 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 8,9 --conf-thres=0.4
# python run/inference_framebyframe.py --breakpoint=1000 --sequence=0930_5 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 9 --conf-thres=0.4
# python run/inference_framebyframe.py --breakpoint=-1 --sequence=0930_6 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 9 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=0930_5 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 2 --conf-thres=0.4
# python run/inference_i_f.py --breakpoint=-1 --sequence=0930_6 --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 2 --conf-thres=0.4

for i in `seq 66`
do
if [ $i -gt 44 ];then 
    # echo $i
# fi
    python run/inference_i_f.py --breakpoint=-1 --sequence=1021_$i --scene=wusi --extrinsics_path ../dataset/new_extrinsics --device 2 --conf-thres=0.4
fi
done