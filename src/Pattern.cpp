#include "SGRS.h"


Pattern::Pattern(std::string path)
	: image(cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE))
{}

double Pattern::match(cv::Mat& obj_mask) {
	int affinity = 0;
	for (int h = 0; h < image.rows; h++)
		for (int w = 0; w < image.rows; w++) {
			if (static_cast<int>(image.at<uchar>(h, w)) != static_cast<int>(obj_mask.at<uchar>(h, w)))
				affinity++;
		}
	return (affinity * 100) / (obj_mask.cols * obj_mask.rows);
}

cv::Mat Pattern::get() const {
	return image;
}

