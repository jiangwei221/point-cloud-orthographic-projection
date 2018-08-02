# point cloud orthographic projection

Simple point cloud orthographic projection, support projection to XY, YZ, ZX planes.

### Introduction

Some time it is useful to get the orthographic projection of a point cloud. Ortho projection can remove the camera distortion. Multiview projection can provide extra information for deep learning tasks.

![ortho-proj-vis](https://imgur.com/mhKxr5d)

### Usage

Run point_cloud_ortho_projector.py for simple visualization.

### Data

For depth image data:

I used the synthstic data from UBC3V dataset. It is a sythetic dataset for human pose estimation.

In the <sample_data> dir, I put several images from UBC3V dataset.

For PCD data:

Loading PCD data is a TO-DO.

### Thanks

UBC3V dataset, link: https://github.com/ashafaei/ubc3v

### TO-DO

Load PCD data.

PCA analysis and OBB.
