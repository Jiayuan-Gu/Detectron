#!/usr/bin/env bash
SRC=/data/coco
DEST=detectron/datasets/data/coco
ln -s $SRC/images/train2014 $DEST/coco_train2014
ln -s $SRC/images/val2014 $DEST/coco_val2014
ln -s $SRC/annotations $DEST/annotations