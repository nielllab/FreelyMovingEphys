#! /bin/bash

srcExt=$1
destExt=$2
srcDir=$3
destDir=$4

for filename in "$srcDir"/*.$srcExt; do
  basePath=${filename%.*}
  baseName=${basePath##*/}
  ffmpeg -i "$filename" -vf yadif=1:-1:0 -c:v libx264 -preset slow -crf 19 -c:a aac -b:a 256k "$destDir"/"$baseName"."$destExt"

done

echo "Conversion from ${srcExt} to ${destExt} complete"
