# Point Cloud Orthographic Projection with Multiviews

Simple point cloud orthographic projection, support projection to XY, YZ, ZX planes.

### Introduction

Some time it is useful to get the orthographic projection of a point cloud. Ortho projection can remove the camera distortion. Multiview projections can provide extra information for deep learning tasks.

![ortho-proj-vis](https://i.imgur.com/mhKxr5d.png)

### Usage

It requires pypcd package, if you are using python3, please install [this version of pypcd](https://github.com/klintan/pypcd). klintan fixed the [cStringIO issue](https://github.com/dimatura/pypcd/pull/9) in the original version.

Run point_cloud_ortho_projector.py for simple visualization.

### Data

For depth image data:

I used the synthstic data from UBC3V dataset. It is a sythetic dataset for human pose estimation.

In the <sample_data> dir, I put several images from UBC3V dataset.

For PCD data:

I used pypcd to load the PCD data.

In the <sample_data> dir, I put bunny PCD file in ascii.

### Thanks

UBC3V dataset, link: https://github.com/ashafaei/ubc3v

### TO-DO

PCA analysis and OBB.
