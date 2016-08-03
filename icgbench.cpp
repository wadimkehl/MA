/*
* Copyright (c) ICG. All rights reserved.
*
* Institute for Computer Graphics and Vision
* Graz University of Technology / Austria
*
* This software is distributed WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
* PURPOSE.  See the above copyright notices for more information.
*
* Author     : Jakob Santner
* EMail      : santner@icg.tugraz.at
*/

#include "icgbench.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <stdexcept>


namespace IcgBench {

  /////////////////////////////////////////////////////////////////////////////
  // Support Functions
  /////////////////////////////////////////////////////////////////////////////
  double computeDiceScore(const LabelImage& a, const LabelImage&b){
    std::map<unsigned int, unsigned int> hist_a, hist_b;
    hist_a = a.histogram();
    hist_b = b.histogram();
    size_t num_labels = std::max(hist_a.size(), hist_b.size());

    double score = 0;
    // for every label
    for (unsigned int label = 0; label < num_labels; label++){
      int area_ab = 0;
      int area_a = 0;
      int area_b = 0;
      for (unsigned int x = 0; x < a.width(); x++){
        for (unsigned int y = 0; y < a.height(); y++){
          if (label == a.get(x,y))
            area_a++;
          if (label == b.get(x,y)){
            area_b++;
            if (label == a.get(x,y))
              area_ab++;
          }
        }
      }
      score += 2.0 * area_ab / (area_a + area_b);
    }
    return score / num_labels;
  }

  /////////////////////////////////////////////////////////////////////////////
  // LabelImage class
  /////////////////////////////////////////////////////////////////////////////  
  LabelImage::LabelImage(unsigned int* src, unsigned int width, unsigned int height)
  {
    labels_ = new unsigned int [width * height];
    width_ = width;
    height_ = height;
    memcpy(labels_, src, width * height * sizeof(unsigned int));
  }

  LabelImage::LabelImage(unsigned int width, unsigned int height)
  {
    labels_ = new unsigned int [width * height];
    width_ = width;
    height_ = height;
  }

  LabelImage::LabelImage(const LabelImage& src)
  {
    labels_ = new unsigned int [src.width_ * src.height_];
    memcpy(labels_, src.labels_, src.width_ * src.height_ * sizeof(unsigned int));
    width_ = src.width_;
    height_ = src.height_;
  } 

  LabelImage::~LabelImage()
  {
    if (labels_)
      delete[] labels_;
  }

  void LabelImage::set(unsigned int value, unsigned int x, unsigned int y)
  {
    if (y >= height_ || x >= width_)
      throw(std::runtime_error("index exceeds label image dimensions"));
    labels_[y * width_ + x] = value;
  }

  unsigned int LabelImage::get(unsigned int x, unsigned int y) const
  {
    if (y >= height_ || x >= width_)
      throw(std::runtime_error("index exceeds label image dimensions"));
    return labels_[y * width_ + x];
  }    

  std::map<unsigned int, unsigned int> LabelImage::histogram() const
  {
    std::map<unsigned int, unsigned int> h;
    std::map<unsigned int, unsigned int>::iterator it;
    // returns the histogram over the labels      
    for (size_t i = 0; i < width_*height_; i++){
      it = h.find(labels_[i]);
      if (it != h.end())
        it->second++;
      else
        h[labels_[i]] = 1;
    }
    return h;
  } 

  /////////////////////////////////////////////////////////////////////////////
  // IcgBenchFileIO class
  /////////////////////////////////////////////////////////////////////////////

  IcgBenchFileIO::IcgBenchFileIO()
  {
  }


  IcgBenchFileIO::IcgBenchFileIO(const std::string& filename)
  {
    // try to open file
    std::ifstream file(filename.c_str());
    if (!file.is_open())
      throw(std::runtime_error("could not open file"));

    // parse header
    char char_buffer[512];
    file.getline(char_buffer,512);
    if (strcmp(char_buffer, "ICGBENCH GROUNDTRUTH FILE"))
      throw(std::runtime_error("file is not recognized as valid ground truth file"));

    std::string string_buffer;
    file >> string_buffer;
    if (strcmp(string_buffer.c_str(), "USER:"))
      throw(std::runtime_error("file is not recognized as valid ground truth file"));
    file >> user_name_;
    file >> string_buffer;
    if (strcmp(string_buffer.c_str(), "IMAGE:"))
      throw(std::runtime_error("file is not recognized as valid ground truth file"));
    file >> file_name_;
    file >> string_buffer;
    if (strcmp(string_buffer.c_str(), "#LABELS:"))
      throw(std::runtime_error("file is not recognized as valid ground truth file"));
    file >> num_labels_;
    unsigned int width, height;
    file >> string_buffer;
    if (strcmp(string_buffer.c_str(), "WIDTH:"))
      throw(std::runtime_error("file is not recognized as valid ground truth file"));
    file >> width;
    file >> string_buffer;
    if (strcmp(string_buffer.c_str(), "HEIGHT:"))
      throw(std::runtime_error("file is not recognized as valid ground truth file"));
    file >> height;

    // create buffers
    labels_ = new LabelImage(width, height);

    // parse labels
    file >> string_buffer;
    if (strcmp(string_buffer.c_str(), "LABELS:"))
      throw(std::runtime_error("file is not recognized as valid ground truth file"));
    unsigned int pos = 0, label, length; 
    while (1) {
      file >> label;
      file >> length;
      for (unsigned int i = pos; i < pos + length; i++){
        size_t y = (size_t) (i / width);
        size_t x = i - y * width;
        labels_->set(label, x, y);
      }
      pos = pos + length;
      if (pos >= width * height)
        break;
    }
    // parse seeds to image structure
    int* seeds = new int[width * height];
    file >> string_buffer;
    if (strcmp(string_buffer.c_str(), "SEEDS:"))
      throw(std::runtime_error("file is not recognized as valid ground truth file"));
    pos = 0;
    while (1) {
      file >> label;
      file >> length;
      for (unsigned int i = pos; i < pos + length; i++)
        seeds[i] = label;
      pos = pos + length;
      if (pos >= width * height)
        break;
    }
    // fill vector with seeds and delete image structure
    for (unsigned int x = 0; x < width; x++){
      for (unsigned int y = 0; y < height; y++){
        int my_label = seeds[y * width + x];
        if (my_label > 0){
          Seed my_seed;
          my_seed.x = x;
          my_seed.y = y;
          my_seed.label = my_label - 1;
          seeds_.push_back(my_seed);
        }
      }
    }
    delete[] seeds;    
  }

  IcgBenchFileIO::~IcgBenchFileIO()
  {
    delete labels_;
  }  

}; // namespace IcgBench {
