#!/bin/bash
# This script creates a symbolic link for the datasets.
# It should be run from the root of the OpenStereo project.

set -e

cd /fdisk1/Codes/OmniDepth

# Create the data directory if it doesn't exist
mkdir -p datasets

# Create the symbolic link
if [ -L "datasets/FSD" ]; then
    echo "Symlink datasets/FSD already exists."
else
    rm -rf datasets/FSD
    ln -s /fdisk1/FSD/ datasets/FSD
    echo "Symlink created for FSD dataset at datasets/FSD"
fi

mkdir -p datasets/KITTI

# Create the symbolic link
if [ -L "datasets/KITTI/KITTI_2012" ]; then
    echo "Symlink datasets/KITTI/KITTI_2012 already exists."
else
    rm -rf datasets/KITTI/KITTI_2012
    ln -s /fdisk1/KITTI12/ datasets/KITTI/KITTI_2012
    echo "Symlink created for KITTI_2012 dataset at datasets/KITTI/KITTI_2012"
fi

# Create the symbolic link
if [ -L "datasets/KITTI/KITTI_2015" ]; then
    echo "Symlink datasets/KITTI/KITTI_2015 already exists."
else
    rm -rf datasets/KITTI/KITTI_2015
    ln -s /fdisk1/KITTI15/ datasets/KITTI/KITTI_2015
    echo "Symlink created for KITTI_2015 dataset at datasets/KITTI/KITTI_2015"
fi

# Create the symbolic link
if [ -L "datasets/Middlebury" ]; then
    echo "Symlink datasets/Middlebury already exists."
else
    rm -rf datasets/Middlebury
    ln -s /fdisk1/Middlebury/ datasets/Middlebury
    echo "Symlink created for Middlebury dataset at datasets/Middlebury"
fi

if [ -L "datasets/SintelStereo" ]; then
    echo "Symlink datasets/SintelStereo already exists."
else
    rm -rf datasets/SintelStereo
    ln -s /fdisk1/Sintel/ datasets/SintelStereo
    echo "Symlink created for SintelStereo dataset at datasets/SintelStereo"
fi

if [ -L "datasets/CREStereo" ]; then
    echo "Symlink datasets/CREStereo already exists."
else
    rm -rf datasets/CREStereo
    ln -s /fdisk1/CREStereo/ datasets/CREStereo
    echo "Symlink created for CREStereo dataset at datasets/CREStereo"
fi

if [ -L "datasets/FallingThings" ]; then
    echo "Symlink datasets/FallingThings already exists."
else
    rm -rf datasets/FallingThings
    ln -s /fdisk1/FallingThings/ datasets/FallingThings
    echo "Symlink created for FallingThings dataset at datasets/FallingThings"
fi

if [ -L "datasets/InStereo2K" ]; then
    echo "Symlink datasets/InStereo2K already exists."
else
    rm -rf datasets/InStereo2K
    ln -s /fdisk1/InStereo2K/ datasets/InStereo2K
    echo "Symlink created for InStereo2K dataset at datasets/InStereo2K"
fi

if [ -L "datasets/VKITTI2" ]; then
    echo "Symlink datasets/VKITTI2 already exists."
else
    rm -rf datasets/VKITTI2
    ln -s /fdisk1/VirtualKitti2/ datasets/VKITTI2
    echo "Symlink created for VirtualKitti2 dataset at datasets/VKITTI2"
fi

if [ -L "datasets/ETH3D" ]; then
    echo "Symlink datasets/ETH3D already exists."
else
    rm -rf datasets/ETH3D
    ln -s /fdisk1/ETH3D/ datasets/ETH3D
    echo "Symlink created for ETH3D dataset at datasets/ETH3D"
fi

if [ -L "datasets/SceneFlow" ]; then
    echo "Symlink datasets/SceneFlow already exists."
else
    rm -rf datasets/SceneFlow
    ln -s /fdisk1/Scene\ Flow/ datasets/SceneFlow
    echo "Symlink created for SceneFlow dataset at datasets/SceneFlow"
fi