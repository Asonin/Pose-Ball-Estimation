数据位置：
    output/wusi/......
    
每一个日期标签（比如1009_2）下面有很多个sequence（如1），里面是一串连续的数据

Define: 
	n == 该片段的总帧数
	x == (该片段总人数,目前默认为5)
ball_pos.shape == (n, 3)
pose_id.shape == (n, x)
读取了 poses.npy 后需要先reshape(-1,x,15,3)
Then poses.shape == (n, x, 15, 3), 每一组poses的id对应关系是和同一帧的pose_id对应的，取用时注意下标设置。
	
