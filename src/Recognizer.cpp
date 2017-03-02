#include "SGRS.h"


Recognizer::Recognizer() {
	subtractor = cv::createBackgroundSubtractorMOG2(500, 16.0, false);
	kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2), cv::Point(1, 1));
}

Recognizer::~Recognizer() {
	for (auto& val : patterns)
		delete val;
}

void Recognizer::binary_mask(cv::Mat& frame, cv::Mat& bin_mask) {
	subtractor->apply(frame, bin_mask);
	erode(bin_mask, bin_mask, kernel, cv::Point(-1, -1), 5); //6
	dilate(bin_mask, bin_mask, kernel, cv::Point(-1, -1), 1);
}

bool Recognizer::empty_mask(cv::Mat& clipped_obj) {
	unsigned p = 0;
	// проверяем все точки маски
	for (int h = 0; h < clipped_obj.rows; h++)
		for (int w = 0; w < clipped_obj.cols; w++)
			p += static_cast<int>(clipped_obj.at<uchar>(h, w));
	// если количество точек, которыми представлен объект меньше допустимого порога
	// считаем, что маска объект не содержит
	return (p <= SGRS::THRESHOLD);
}

double Recognizer::cmd_by_index(cv::Mat obj, unsigned idx) {
	return patterns[idx]->match(obj);
}

double Recognizer::cmd_in_range(cv::Mat obj, int start, int end) {
	if (start > end) return 0.f;

	double* aff = new double[end - start];
	int current = start;
	for (int i = 0; i < (end - start); i++) {
		aff[i] = cmd_by_index(obj, current);
	}
	double best = 100.f;
	for (int i = 0; i < (end - start); i++)
		if (aff[i] < best)
			best = aff[i];
	delete[] aff;
	return best;
}

Command Recognizer::recognize(cv::Mat& clipped_obj) {
	if (empty_mask(clipped_obj))
		return Command::NOTHING;

	std::vector<double> affinity(Recognizer::UNIQUE_CMDS);

	affinity[0] = cmd_in_range(clipped_obj, 0, 10);
	affinity[1] = cmd_in_range(clipped_obj, 11, 16);
	affinity[2] = cmd_in_range(clipped_obj, 17, 22);
	affinity[3] = cmd_by_index(clipped_obj, 23);
	affinity[4] = cmd_by_index(clipped_obj, 24);

	double best_aff = 100; //100%
	Command cmd = Command::NOTHING;

	if (affinity[0] < best_aff) {
		best_aff = affinity[0]; // "|"
		cmd = Command::VERTICAL;
	}
	if (affinity[1] < best_aff) {
		best_aff = affinity[1]; // "|--"
		cmd = Command::HORIZONTAL_NEXT;
	}
	if (affinity[2] < best_aff) {
		best_aff = affinity[2]; // "--|"
		cmd = Command::HORIZONTAL_PREV;
	}
	if (affinity[3] < best_aff) {
		best_aff = affinity[3]; // "\"
		cmd = Command::DIAGONAL_L;
	}
	if (affinity[4] < best_aff) {
		best_aff = affinity[4]; // "/"
		cmd = Command::DIAGONAL_R;
	}

	if (command_buffer.size() >= SGRS::CMD_BUFFER_SIZE) {
		Command c = command_buffer[0];
		for (auto buff_entry : command_buffer)
			if (buff_entry != c) {
				command_buffer.clear();
				return Command::NOTHING;
			}
		command_buffer.clear();
		return c;
	}
	else {
		command_buffer.push_back(cmd);
		return Command::NOTHING;
	}
}

void Recognizer::add_pattern(Pattern* p) {
	patterns.push_back(p);
}

Command Recognizer::last_command() {
	return prev_cmd;
}

void Recognizer::last_command(Command c) {
	prev_cmd = c;
}
