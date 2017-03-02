#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>

#include "SGRS.h" // Object-oriented Gesture Recognition System

int main() {

	SGRS sgrs(CV_CAP_ANY);

	sgrs.add_patterns_population("patterns/vert_", 11);
	sgrs.add_patterns_population("patterns/horiz_next_", 6);
	sgrs.add_patterns_population("patterns/horiz_prev_", 6);
	sgrs.add_pattern(new Pattern("patterns/lh_diag.bmp"));
	sgrs.add_pattern(new Pattern("patterns/rh_diag.bmp"));

	cvNamedWindow("Frame", 1);
	cvNamedWindow("Object", 1);

	while (sgrs.read()) {

		cv::Mat fgm = sgrs.foregr_mask();
		cv::Mat om = sgrs.object_mask();
		
		switch (sgrs.recognize_obj()) {
		case Command::VERTICAL:
			std::cout << "Command: VERTICAL" << std::endl;
			break;
		case Command::HORIZONTAL_NEXT:
			std::cout << "Command: HORIZONTAL_NEXT" << std::endl;
			break;
		case Command::HORIZONTAL_PREV:
			std::cout << "Command: HORIZONTAL_PREV" << std::endl;
			break;
		case Command::DIAGONAL_L:
			std::cout << "Command: DIAGONAL_L" << std::endl;
			break;
		case Command::DIAGONAL_R:
			std::cout << "Command: DIAGONAL_R" << std::endl;
			break;
		default: //Command::NOTHING
			break;
		}

		cv::imshow("Frame", fgm);
		cv::imshow("Object", om);

		// клавиша "Esc" завершает программу
		char key = cv::waitKey(30);
		if (key == 27) break;
	}
	return 0;
}
