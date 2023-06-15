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

# future

- 可以结合Neural Supersampling for Real-time Rendering、High-Quality Supersampling via Mask-reinforced Deep Learning for Real-time Rendering，从而获得16*4=64倍的像素提升！

具体是：
1.渲染512*256的1/4的像素
2.用后者，重建为512*256
3.用前者，放大为512*256*16的分辨率



- color div, multi albedo?

