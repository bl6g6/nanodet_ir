//
// Create by RangiLyu
// 2020 / 10 / 2
//

#include "nanodet.h"
#include <chrono>
#include <net.h>

namespace {
	int input_size[2] = { 320, 320 };
	int num_class = 4;
	int reg_max = 7;
	const int color_list[1][1] =
	{
		//{255 ,255 ,255}, //bg
		{ 255 },
	};


	inline float fast_exp(float x)
	{
		union {
			uint32_t i;
			float f;
		} v{};
		v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
		return v.f;
	}


	inline float sigmoid(float x)
	{
		return 1.0f / (1.0f + fast_exp(-x));
	}


	template<typename _Tp>
	int activation_function_softmax(const _Tp* src, _Tp* dst, int length)
	{
		const _Tp alpha = *std::max_element(src, src + length);
		_Tp denominator{ 0 };

		for (int i = 0; i < length; ++i) {
			dst[i] = fast_exp(src[i] - alpha);
			denominator += dst[i];
		}

		for (int i = 0; i < length; ++i) {
			dst[i] /= denominator;
		}

		return 0;
	}


	nanodet::BoxInfo disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride)
	{
		float ct_x = (x + 0.5) * stride;
		float ct_y = (y + 0.5) * stride;
		std::vector<float> dis_pred;
		dis_pred.resize(4);
		for (int i = 0; i < 4; i++)
		{
			float dis = 0;
			float* dis_after_sm = new float[reg_max + 1];
			activation_function_softmax(dfl_det + i * (reg_max + 1), dis_after_sm, reg_max + 1);
			for (int j = 0; j < reg_max + 1; j++)
			{
				dis += j * dis_after_sm[j];
			}
			dis *= stride;
			//std::cout << "dis:" << dis << std::endl;
			dis_pred[i] = dis;
			delete[] dis_after_sm;
		}
		float xmin = (std::max)(ct_x - dis_pred[0], .0f);
		float ymin = (std::max)(ct_y - dis_pred[1], .0f);
		float xmax = (std::min)(ct_x + dis_pred[2], (float)input_size[0]);
		float ymax = (std::min)(ct_y + dis_pred[3], (float)input_size[1]);

		//std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
		return nanodet::BoxInfo{ xmin, ymin, xmax, ymax, score, label };
	}


	void decode_infer(ncnn::Mat& cls_pred, ncnn::Mat& dis_pred, int stride, float threshold, std::vector<std::vector<nanodet::BoxInfo>>& results)
	{
		int feature_h = input_size[1] / stride;
		int feature_w = input_size[0] / stride;

		//cv::Mat debug_heatmap = cv::Mat(feature_h, feature_w, CV_8UC3);
		for (int idx = 0; idx < feature_h * feature_w; idx++)
		{
			const float* scores = cls_pred.row(idx);
			int row = idx / feature_w;
			int col = idx % feature_w;
			float score = 0;
			int cur_label = 0;
			for (int label = 0; label < num_class; label++)
			{
				if (scores[label] > score)
				{
					score = scores[label];
					cur_label = label;
				}
			}
			if (score > threshold)
			{
				//std::cout << "label:" << cur_label << " score:" << score << std::endl;
				const float* bbox_pred = dis_pred.row(idx);
				results[cur_label].push_back(disPred2Bbox(bbox_pred, cur_label, score, col, row, stride));
				//debug_heatmap.at<cv::Vec3b>(row, col)[0] = 255;
				//cv::imshow("debug", debug_heatmap);
			}
		}
	}


	void nms(std::vector<nanodet::BoxInfo>& input_boxes, float NMS_THRESH)
	{
		std::sort(input_boxes.begin(), input_boxes.end(), [](nanodet::BoxInfo a, nanodet::BoxInfo b) { return a.score > b.score; });
		std::vector<float> vArea(input_boxes.size());
		for (int i = 0; i < int(input_boxes.size()); ++i) {
			vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
				* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
		}
		for (int i = 0; i < int(input_boxes.size()); ++i) {
			for (int j = i + 1; j < int(input_boxes.size());) {
				float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
				float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
				float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
				float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
				float w = (std::max)(float(0), xx2 - xx1 + 1);
				float h = (std::max)(float(0), yy2 - yy1 + 1);
				float inter = w * h;
				float ovr = inter / (vArea[i] + vArea[j] - inter);
				if (ovr >= NMS_THRESH) {
					input_boxes.erase(input_boxes.begin() + j);
					vArea.erase(vArea.begin() + j);
				}
				else {
					j++;
				}
			}
		}
	}

	void global_nms(std::vector<nanodet::BoxInfo>& input_boxes, float G_NMS_THRESH)
	{
		std::sort(input_boxes.begin(), input_boxes.end(), [](nanodet::BoxInfo a, nanodet::BoxInfo b) { return a.score > b.score; });
		std::vector<float> vArea(input_boxes.size());
		for (int i = 0; i < int(input_boxes.size()); ++i) {
			vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
				* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
		}
		for (int i = 0; i < int(input_boxes.size()); ++i) {
			for (int j = i + 1; j < int(input_boxes.size());) {
				float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
				float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
				float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
				float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
				float w = (std::max)(float(0), xx2 - xx1 + 1);
				float h = (std::max)(float(0), yy2 - yy1 + 1);
				float inter = w * h;
				// float ovr = inter / (vArea[i] + vArea[j] - inter);
				float ovr2 = inter / vArea[i];    // 增加更加严格的覆盖率计算方式.
				float ovr3 = inter / vArea[j];
				if ((ovr2 > ovr3 ? ovr2: ovr3) >= G_NMS_THRESH) {      // 此处的G_NMS_THRESH是可以根据需要进行改动的.
					input_boxes.erase(input_boxes.begin() + j);
					vArea.erase(vArea.begin() + j);
				}
				else {
					j++;
				}
			}
		}
	}
}

namespace nanodet{
    bool NanoDet::hasGPU = false;
    ncnn::Net* Net;

    NanoDet::NanoDet()
    {
        Net = new ncnn::Net();
		if (NanoDet::hasGPU)
		{
#if NCNN_VULKAN
			if (ncnn::create_gpu_instance())
			{
				std::cerr << "create gpu instance fail!" << std::endl;
			}
			Net->opt.use_vulkan_compute = true;
#endif
		}
    }


    NanoDet::~NanoDet()
    {
		if (NanoDet::hasGPU)
		{
#if NCNN_VULKAN
			ncnn::destroy_gpu_instance();
#endif
		}
        delete Net;
    }


    bool NanoDet::init(const std::string& bin_path, const std::string& param_path)
    {
//    #if NCNN_VULKAN
//        this->hasGPU = ncnn::get_gpu_count() > 0;
//    #endif
//        this->Net->opt.use_vulkan_compute = this->hasGPU && useGPU;
//        this->Net->opt.use_fp16_arithmetic = true;
        // this->Net->opt.use_int8_inference = true;
        // init param
        int ret =Net->load_param(param_path.c_str());
        if (ret != 0) {
            std::cerr << "load model param failed" << std::endl;
            return false;
        }
        // init bin
        ret = Net->load_model(bin_path.c_str());
        if (ret != 0) {
            std::cerr << "load model bin failed" << std::endl;
            return false;
        }
        return true;
    }


    std::vector<BoxInfo> NanoDet::detect(const unsigned char *pixels, const int& width, const int& height, const int& target_size,
		float score_threshold, float nms_threshold, bool useGPU)
    {
		ncnn::Mat input;
		object_rect effect_roi;

		{
			int dst_w = target_size;
			int dst_h = target_size;

			float ratio_src = width * 1.0 / height;
			float ratio_dst = dst_w * 1.0 / dst_h;

			if (ratio_src > ratio_dst) {
				dst_w = dst_w;
				dst_h = floor((dst_w * 1.0 / width) * height);
			}
			else if (ratio_src < ratio_dst) {
				dst_h = dst_h;
				dst_w = floor((dst_h * 1.0 / height) * width);
			}
			input = ncnn::Mat::from_pixels_resize(pixels, ncnn::Mat::PIXEL_GRAY, width, height, dst_w, dst_h);

			int pad_top = (target_size - dst_h) / 2;
			int pad_bottom = target_size - dst_h - pad_top;

			int pad_left = (target_size - dst_w) / 2;
			int pad_right = target_size - dst_w - pad_left;

			effect_roi.x = pad_left;
			effect_roi.y = pad_top;
			effect_roi.width = dst_w;
			effect_roi.height = dst_h;

			ncnn::copy_make_border(input, input, pad_top, pad_bottom, pad_left, pad_right, ncnn::BorderType::BORDER_CONSTANT, 0);

            const float mean_vals[1] = {129.75343168f}; 
            const float norm_vals[1] = { 0.01387678693f };

			input.substract_mean_normalize(mean_vals, norm_vals);
		}
        //cv::Mat image(height, width, CV_8UC1, (void *)pixels);
        //preprocess(image, input, target_size, effect_roi);

        int dst_w = effect_roi.width;
        int dst_h = effect_roi.height;
        float width_ratio = (float)width / (float)dst_w;
        float height_ratio = (float)height / (float)dst_h;

        auto ex = Net->create_extractor();
        ex.set_light_mode(true);
        ex.set_num_threads(4);
    #if NCNN_VULKAN
		if (NanoDet::hasGPU)
			ex.set_vulkan_compute(this->hasGPU);
    #endif
        ex.input("input.1", input);

        std::vector<std::vector<BoxInfo>> results;
        results.resize(num_class);

        for (const auto& head_info : this->heads_info)
        {
            ncnn::Mat dis_pred;
            ncnn::Mat cls_pred;
            ex.extract(head_info.dis_layer.c_str(), dis_pred);
            ex.extract(head_info.cls_layer.c_str(), cls_pred); 
            // std::cout << "c:" << cls_pred.c << " h:" << cls_pred.h <<" w:" <<cls_pred.w <<std::endl;

            decode_infer(cls_pred, dis_pred, head_info.stride, score_threshold, results);
        }

        std::vector<BoxInfo> dets;
        for (int i = 0; i < (int)results.size(); i++)
        {
            nms(results[i], nms_threshold);
            
            for (auto box : results[i])
            {
                BoxInfo& bbox = box;
                bbox.x1 = (box.x1 - effect_roi.x) * width_ratio;
                bbox.y1 = (box.y1 - effect_roi.y) * height_ratio;
                bbox.x2 = (box.x2 - effect_roi.x) * width_ratio;
                bbox.y2 = (box.y2 - effect_roi.y) * height_ratio;
                dets.push_back(bbox);
            }
        }
		// 再进行一次全局nms过滤，不区分类别.剔除一个人体有两个不同类别框的情况.
		float G_NMS_THRESH = 0.8;
        // auto start = std::chrono::steady_clock::now();
		global_nms(dets,  G_NMS_THRESH);
        // auto end = std::chrono::steady_clock::now();
        // auto cost_time = std::chrono::duration<double, std::milli>(end - start).count();
        // printf("global_nms  time: %7.2f ms. \n", cost_time);
        return dets;
    }
}