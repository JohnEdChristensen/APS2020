#!/bin/bash
 
for file in `find $1/ -type f -name "*.mp4"`; do
#for file in `find $1/ -type f -`; do
     
    # check to see if a poster already exists
     
    if [ ! -e "${file/.mp4}.jpg" ]
    then
        # make a poster
        #echo $file
        ffmpeg -i $file -vf "select=eq(n\,0)" -q:v 3 ${file/.mp4/}.jpg
    fi
done
