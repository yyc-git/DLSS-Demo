import { getInputTensor } from "./common/utils"
import view_current_frame_img_path from './dataset/View/74.png'
import view_last_frame_1_img_path from './dataset/View/75.png'
import view_last_frame_2_img_path from './dataset/View/76.png'
import view_last_frame_3_img_path from './dataset/View/77.png'
import view_last_frame_4_img_path from './dataset/View/78.png'
import view_last_frame_5_img_path from './dataset/View/79.png'
import depth_current_frame_img_path from './dataset/Depth/74.png'
import depth_last_frame_1_img_path from './dataset/Depth/75.png'
import depth_last_frame_2_img_path from './dataset/Depth/76.png'
import depth_last_frame_3_img_path from './dataset/Depth/77.png'
import depth_last_frame_4_img_path from './dataset/Depth/78.png'
import depth_last_frame_5_img_path from './dataset/Depth/79.png'

let _loadImage = (url) => {
    let image = new Image()
    image.src = url

    return new Promise((resolve) => {
        image.onload = () => {
            resolve(image)
        }
    })
}

let _getInputTensor = async (img_path, inputDimensions,) => {
    return getInputTensor(await _loadImage(img_path), {
        inputDimensions: inputDimensions,
        scaledFlag: true,
        inputLayout: 'nchw',
        norm: true
    })
}

export let loadInputs = async ([width, height]) => {
    return [
        [
            view_current_frame_img_path,
            view_last_frame_1_img_path,
            view_last_frame_2_img_path,
            view_last_frame_3_img_path,
            view_last_frame_4_img_path,
            view_last_frame_5_img_path,
        ].map(async img_path => {
            return await _getInputTensor(img_path, [1, 3, height, width])
        }),
        [
            depth_current_frame_img_path,
            depth_last_frame_1_img_path,
            depth_last_frame_2_img_path,
            depth_last_frame_3_img_path,
            depth_last_frame_4_img_path,
            depth_last_frame_5_img_path,
        ].map(async img_path => {
            return await _getInputTensor(img_path, [1, 1, height, width])
        })
    ]
}