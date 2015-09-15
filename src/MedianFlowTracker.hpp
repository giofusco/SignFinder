/**
 * @file PatchTracker.hpp
 *
 *  Created on: @date Mar 12, 2013
 *      Author: @author ender tekin
 */

#ifndef MEDIANFLOWTRACKER_HPP_
#define MEDIANFLOWTRACKER_HPP_

#include <cstdint>
#include <opencv2/core/core.hpp>

typedef cv::Mat_<std::uint8_t> MatUint8;

/// Median flow tracker.
/// Based on http://www3.ee.surrey.ac.uk/CVSSP/Publications/papers/Kalal-ICPR-2010.pdf
/// @param[in] loc location of the object in the previous frame
/// @param[in] prevImg grayscale representation of the previous frame
/// @param[in] currentImg grayscale representation of the current frame
/// @return estimated location of the object in currentImg. If object is lost, returns cv::Rect()
/// TODO: See if we can do away with the whole frame needed in prevImg, and just use the previous patch instead
cv::Rect trackMedianFlow(const cv::Rect& loc, const MatUint8& prevImg, const MatUint8& currentImg);

#endif /* MEDIANFLOWTRACKER_HPP_ */
