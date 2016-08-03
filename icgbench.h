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

#ifndef ICGBENCH_H_
#define ICGBENCH_H_

#include <string>
#include <vector>
#include <map>

namespace IcgBench{

  /////////////////////////////////////////////////////////////////////////////
  // LabelImage class
  /////////////////////////////////////////////////////////////////////////////
  class LabelImage{
  public:
    LabelImage(unsigned int* src, unsigned int width, unsigned int height);
    LabelImage(unsigned int width, unsigned int height);
    LabelImage(const LabelImage& src);
    ~LabelImage();

    void set(unsigned int value, unsigned int x, unsigned int y);
    unsigned int get(unsigned int x, unsigned int y) const;
    unsigned int width() const{return width_;};
    unsigned int height() const{return height_;};
    std::map<unsigned int, unsigned int> histogram() const;

  private:
    unsigned int* labels_;
    unsigned int width_, height_;
  };

  /////////////////////////////////////////////////////////////////////////////
  // Support Functions
  /////////////////////////////////////////////////////////////////////////////
  double computeDiceScore(const LabelImage& a, const LabelImage&b);

  /////////////////////////////////////////////////////////////////////////////
  // Seed struct
  /////////////////////////////////////////////////////////////////////////////
  typedef struct{
    unsigned int x;
    unsigned int y;
    unsigned int label;
  } Seed;

  /////////////////////////////////////////////////////////////////////////////
  // IcgBenchFileIO class
  /////////////////////////////////////////////////////////////////////////////
  class IcgBenchFileIO{
  public:
    IcgBenchFileIO(const std::string& filename);
    ~IcgBenchFileIO();  

    LabelImage* getLabels(){return new LabelImage(*labels_);};
    unsigned int getNumLabels(){return num_labels_;};
    std::vector<Seed> getSeeds(){return seeds_;};
    std::string getFileName(){return file_name_;};
    std::string getUserName(){return user_name_;};

    IcgBenchFileIO();
    IcgBenchFileIO(const IcgBenchFileIO&);

    LabelImage* labels_;    
    unsigned int num_labels_;
    std::vector<Seed> seeds_;
    std::string file_name_, user_name_; 
  };
}


#endif // ICGBENCH_H_
