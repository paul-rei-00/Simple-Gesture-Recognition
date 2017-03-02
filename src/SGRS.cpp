#include "SGRS.h"


SGRS::SGRS(int camera)
	: capture(camera),
	  recognizer(new Recognizer())
{
	capture.set(cv::CAP_PROP_FRAME_WIDTH, 320);
	capture.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
}

SGRS::SGRS(int camera, Recognizer* rec)
	: capture(camera),
	  recognizer(rec)
{
	capture.set(cv::CAP_PROP_FRAME_WIDTH, 320);
	capture.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
}

SGRS::~SGRS() {
	if (capture.isOpened())
		capture.release();

	delete recognizer;
}

bool SGRS::read() {
	bool result = capture.read(frame);
	cv::resize(frame, frame, cv::Size(320, 240));
	return result;
}

cv::Mat& SGRS::raw_frame() {
	return frame;
}

cv::Mat& SGRS::foregr_mask() {
	recognizer->binary_mask(frame, bin_mask);
	return bin_mask;
}

cv::Mat& SGRS::object_mask() {
	clip_object();
	return clipped_obj;
}

Command SGRS::recognize_obj() {
	foregr_mask();
	object_mask();
	Command cmd = recognizer->recognize(clipped_obj);
	last_command(cmd);
	return cmd;
}

void SGRS::add_pattern(Pattern* p) {
	recognizer->add_pattern(p);
}

void SGRS::add_patterns_population(std::string dest, unsigned amount) {
	for (int i = 0; i < amount; i++) {
		std::string name = dest;
		std::string ext = ".bmp";
		name += std::to_string(i) += ext;
		this->add_pattern(new Pattern(name));
	}
}

Command SGRS::last_command() const {
	return recognizer->last_command();
}

void SGRS::last_command(Command c) {
	recognizer->last_command(c);
}

void SGRS::clip_object() {
	if (bin_mask.empty())
		recognizer->binary_mask(frame, bin_mask);

	cv::Rect r = cv::boundingRect(bin_mask);

	if (r.width == 0 && r.height == 0) {
		r.width = CLIPPING_SIZE;
		r.height = CLIPPING_SIZE;
	}

	// коррекция ширины
	if (r.width < r.height) {
		int exp_size = (r.height - r.width) / 2;

		if ((r.x - exp_size) < 0)
			r.x = 0;
		else
			r.x -= exp_size;

		if ((r.width + exp_size) > bin_mask.rows)
			r.width = bin_mask.cols;
		else
			r.width += exp_size;
	}
	// корекция высоты
	else {
		int exp_size = (r.width - r.height) / 2;

		if ((r.y - exp_size) < 0)
			r.y = 0;
		else
			r.y -= exp_size;

		if ((r.height + exp_size) > bin_mask.rows)
			r.height = bin_mask.rows;
		else
			r.height += exp_size;
	}

	// вырезаем обьект и масштабируем его к размерам паттернов
	clipped_obj = bin_mask(r);
	cv::resize(clipped_obj, clipped_obj, cv::Size(CLIPPING_SIZE, CLIPPING_SIZE));
}

