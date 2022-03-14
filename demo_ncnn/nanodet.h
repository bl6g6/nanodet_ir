//
// Create by LiuQinglong
// 2021 / 06 / 11
//

#ifndef NANODET_H
#define NANODET_H

#include <iostream>
#include <vector>

namespace nanodet{

    struct HeadInfo
    {
        std::string cls_layer;
        std::string dis_layer;
        int stride;
    };

    struct BoxInfo 
    {
        float x1;
        float y1;
        float x2;
        float y2;
        float score;
        int label;
    };

    struct object_rect {
        int x;
        int y;
        int width;
        int height;
    };

    class NanoDet
    {
    public:
        NanoDet();

        ~NanoDet();

        static bool hasGPU;

        std::vector<HeadInfo> heads_info{
            // cls_pred|dis_pred|stride
            {"cls_pred_stride_8", "dis_pred_stride_8", 8},
            {"cls_pred_stride_16", "dis_pred_stride_16", 16},
            {"cls_pred_stride_32", "dis_pred_stride_32", 32},
        };
		
        /// nanodet model init
        /// \param bin_path: filepath of model.bin
        /// \param param_path: filepath of model.param
        /// \return init flags, trur - success, false - failure
        bool init(const std::string& bin_path, const std::string& param_path);
		
        /// nanodet detection func
        /// \param pixels: image pixels data
        /// \param width: image width
        /// \param height: image height
        /// \param target_size: target size to detect
        /// \param score_threshold: threshold of classification probability
        /// \param nms_threshold: threshold of Non-maximum suppression
        /// \param useGPU: use gpu
        /// \return detection results  box: x1, y1, x2, y2, score, label.  (label returns int: 0,1,2,3)
        /// 0:standing 
        /// 1:seated  
        /// 2:lying  
        /// 3:other
        std::vector<BoxInfo> detect(
			const unsigned char *pixels,
			const int& width,
			const int& height,
			const int& target_size, 
			float score_threshold = 0.35f, 
			float nms_threshold = 0.4f, 
			bool useGPU = false
		);
    };
}

#endif //NANODET_H
