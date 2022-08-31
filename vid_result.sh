for i in 1 2 3 4 5 6 7 8 9 10 11
do
   ffmpeg -f image2 -i out/0315_2/synced/$i/%08d.jpg -vcodec mpeg4 -b 4M out/0315_2/synced/$i.mp4
done
