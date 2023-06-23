# note

- mip level bias

> For lowresolution input images, we turn off MSAA and adjust mip level
bias for texture sampling to match the selected mip level with the
full resolution images. The mip level bias approach is applied to
reduce prefiltering in the rendered low-resolution images and is
similarly done in existing supersampling algorithms such as TAAU
in Unreal

- train

256 × 256 patch



# 代码解读

"node:"是我加入的注释



5 previous frames

input->motion is optical flow, should convert it to motion vector

not use 256 x 256 patch, directly use whole:(120, 180)?


# future



- color div, multi albedo?



- 可以结合Neural Supersampling for Real-time Rendering、High-Quality Supersampling via Mask-reinforced Deep Learning for Real-time Rendering，从而获得16*4=64倍的像素提升！

具体是：
1.渲染512*256的1/4的像素
2.用后者，重建为512*256
3.用前者，放大为512*256*16的分辨率

- not use motion vector?

refer to:
[Fast Temporal Reprojection without Motion Vectors](https://www.google.com/search?q=Fast+Temporal+Reprojection+without+Motion+Vectors&oq=Fast+Temporal+Reprojection+without+Motion+Vectors&aqs=chrome..69i57.138j0j7&sourceid=chrome&ie=UTF-8)