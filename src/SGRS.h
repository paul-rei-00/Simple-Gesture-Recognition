#ifndef SGRS_H
#define SGRS_H

#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\video.hpp>
#include <iostream>
#include <cmath>
#include <memory>

enum class Command {
	NOTHING		= -1,
	VERTICAL		= 0,
	HORIZONTAL_NEXT = 1,
	HORIZONTAL_PREV = 2,
	DIAGONAL_L		= 3,
	DIAGONAL_R		= 4
};

class Pattern {
	cv::Mat image;

public:
	explicit Pattern(std::string path);
	
	double match(cv::Mat& obj_mask);

	cv::Mat get() const;
};

class Recognizer {
protected:

	static const unsigned UNIQUE_CMDS = 5; // can recognize 5 cmds

	cv::Ptr<cv::BackgroundSubtractor> subtractor;
	cv::Mat kernel;

	std::vector<Pattern*> patterns;

	std::vector<Command> command_buffer;

	Command prev_cmd;

public:
	Recognizer();

	virtual ~Recognizer();

	bool empty_mask(cv::Mat& obj_mask);
	
	virtual void binary_mask(cv::Mat& frame, cv::Mat& bin_mask);

	virtual Command recognize(cv::Mat& clipped_object);

	void add_pattern(Pattern* p);

	Command last_command();

	void last_command(Command c);

	double virtual cmd_in_range(cv::Mat obj, int start, int end);
	
	double virtual cmd_by_index(cv::Mat obj, unsigned idx);
};

class SGRS {

	cv::VideoCapture capture;

	cv::Mat frame, bin_mask, clipped_obj;

	Recognizer* recognizer;

public:
	static const unsigned CLIPPING_SIZE = 16U; // 16px
	
	static const unsigned CMD_BUFFER_SIZE = 10U; // 10 commands

	static const unsigned THRESHOLD = 5U; // 5px

	explicit SGRS(int camera);
	
	explicit SGRS(int camera, Recognizer* rec);
	
	~SGRS();

	bool read();

	cv::Mat& raw_frame();

	cv::Mat& foregr_mask();

	cv::Mat& object_mask();

	Command recognize_obj();

	void add_pattern(Pattern* p);

	void add_patterns_population(std::string dest, unsigned amount);

	Command last_command() const;

	void last_command(Command c);

private:

	void clip_object();
};

#endif
