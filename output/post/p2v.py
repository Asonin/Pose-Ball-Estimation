import os 
import ffmpeg

def postprocess(sequence):

    dir_base = f'/home1/zhuwentao/projects/multi-camera/mvball_proj/output/wusi/{sequence}'
    file = os.listdir(dir_base)
    
    for f in file:
        p1 = dir_base + f'/{f}/3d_vis'
        p2 = dir_base + f'/{f}/image_with_projected_ball_and_poses'
        # print(p1)
        
        pics_3d = p1 + '/test_%8d_3d.jpg'
        
        out_dir_3d = dir_base + f'/{f}/3d.mp4'
        if os.path.exists(out_dir_3d):
            continue
        
        (
            ffmpeg
            .input(pics_3d, framerate=25)
            .output(out_dir_3d)
            .run()
        )
        for i in range(1,12):
            pics_projected = p2 + f'/test_%8d_view_{i}.jpg'
            out_dir_projected = dir_base + f'/{f}/view{i}.mp4'
            (
                ffmpeg
                .input(pics_projected, framerate=25)
                .output(out_dir_projected)
                .run()
            )
        
sequence = input('input sequence to trans p to v')
# serial = input('input serial')
postprocess(sequence)