//
// Created by Giuseppe Coviello on 10/5/23.
//

#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "BYTETracker.h"

int main(int argc, const char **argv) {
  if (argc!=3) {
    std::cerr << "Usage: " << argv[0] << " video_file detections_file" << std::endl;
    return 255;
  }

  cv::VideoCapture vc(argv[1]);
  auto fps = (int) round(vc.get(cv::CAP_PROP_FPS));
  std::cout << "Video FPS: " << fps << std::endl;
  auto delay = (int) round(1000.0/vc.get(cv::CAP_PROP_FPS));

  std::ifstream ifs(argv[2]);
  auto detections = nlohmann::json::parse(ifs);
  BYTETracker tracker(fps);

  auto frame_number = 0;
  cv::Mat frame;
  while (vc.read(frame)) {
    auto objects = detections[std::to_string(frame_number)];
    std::vector<Object> people;
    for (const auto &object : objects) {
      if (object["class"]!="person") {
        continue;
      }
      people.push_back(Object{
          .x=object["x"],
          .y=object["y"],
          .width=object["width"],
          .height=object["height"],
          .prob=object["score"],
      });
    }
    auto tracks = tracker.update(people);
    for (const auto &track : tracks) {
      auto color = tracker.get_color(track.track_id);
      cv::rectangle(frame, cv::Rect(track.tlwh[0], track.tlwh[1], track.tlwh[2], track.tlwh[3]),
                    color, 2);
    }
    cv::imshow("tracked people", frame);
    cv::waitKey(delay);
    frame_number += 1;
  }
}