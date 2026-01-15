#!/usr/bin/env bash

url="https://www.kaggle.com/api/v1/datasets/download/vijayabhaskar96/pascal-voc-2007-and-2012"
outdir="./dataset/voc-datasets"
tempzipfile="./pascal-voc-2007-and-2012.zip"

# download
curl -L -o $tempzipfile $url
if [ $? -ne 0 ]; then
    echo "Error: Failed to download the dataset."
    exit 1
fi

# unzip
mkdir -p $outdir
echo "Unzipping dataset to $outdir.."
unzip -q $tempzipfile -d $outdir

if [ $? -ne 0 ]; then
    echo "Error: Failed to unzip the dataset."
    exit 1
fi

# clean uo
rm -v $tempzipfile
mv -v $outdir/VOCdevkit/* $outdir/
rmdir -v $outdir/VOCdevkit

echo Done.