/* All rights reserved.
 *
 *  
 * Software License Agreement (BSD License 2.0)
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *  * Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE. 
 * 
 * Copyright Singapore University of Technology and Design (2023)
 * Author: chhathuranga@gmail.com
 */

#include <chrono>
#include <string>
#include <list>
#include <mutex>
#include <map>
#include<string>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/bool.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "visualization_msgs/msg/marker_array.hpp"

#include "mmpf_explore/msg/exploration_state.hpp"
#include "tf2_ros/transform_broadcaster.h"




#define DEBUG
#define DETECT_END

namespace exploration{
  class MMPF : public rclcpp::Node
  {
    public:
      enum VisualizationType 
      { 
        POINTS = 0,
        LINE = 1,
        SPHERES = 2
      };

      struct Pixel
      {
        int x = 0;
        int y = 0;

        Pixel() {};
        Pixel(int x, int y): x(x), y(y) {};
        inline bool operator==(const Pixel& p) { return ((p.x == x) && (p.y == y)); };
      };

      struct Cluster
      {
        int x = 0;    // Cluster center x
        int y = 0;    // Cluster center y
        int size = 0; // Number of points in cluster

        Cluster() {};
        Cluster(int x, int y, int size): x(x), y(y), size(size) {};
        Cluster(Pixel point, int size=1): x(point.x), y(point.y), size(size) {};
        Pixel center() { return Pixel(x, y); };
      };

      MMPF();
 
      
    private:
      void map_callback(const nav_msgs::msg::OccupancyGrid msg);
      void costmap_callback(const nav_msgs::msg::OccupancyGrid msg);
      void enable_exploration_callback(const std_msgs::msg::Bool msg);

      void set_costmap_data(nav_msgs::msg::OccupancyGrid costmapData);
      nav_msgs::msg::OccupancyGrid get_costmap_data();
      void set_map_data(nav_msgs::msg::OccupancyGrid mapData);
      nav_msgs::msg::OccupancyGrid get_map_data();
      bool map_data_available(void);

      void explore();
      bool get_ros_parameters(void);
      bool get_transform(std::string target_frame, std::string source_frame, geometry_msgs::msg::TransformStamped &transform);
      void publish_exploration_state(void);
      visualization_msgs::msg::Marker create_visualization_msg(int type, double lifetime = 1.0);
      void find_frontiers(nav_msgs::msg::OccupancyGrid mapData, nav_msgs::msg::OccupancyGrid costmapData, std::vector<Pixel> &targets);
      std::vector<MMPF::Cluster> cluster_2D(std::vector<MMPF::Pixel> points, int proximity_threshold = 3);
      bool create_potential_map(nav_msgs::msg::OccupancyGrid mapData, 
                                Pixel source_point, 
                                std::vector<int> &potential_map, 
                                int potential_step = 1);
      inline void process_pixel_potential(Pixel source_pixel, 
                                          Pixel target_pixel, 
                                          std::vector<int> &map, 
                                          std::vector<int> &potential_map, 
                                          int map_width, 
                                          std::vector<Pixel> &discovered_pixels,
                                          int unit_potential);
      std::vector<MMPF::Pixel> calculate_exploration_path(MMPF::Pixel current_location,
                                                          std::vector<MMPF::Cluster> target_clusters,
                                                          std::vector<std::vector<int>> &cluster_potential_maps,
                                                          std::vector<geometry_msgs::msg::Transform> peer_robot_transforms,
                                                          nav_msgs::msg::MapMetaData map_info);

      // Parameters
      std::string map_topic_, costmap_topic_; 
      std::string robot_base_frame_, map_frame_, robot_frame_prefix_;
      float rate_;
      int robot_count_;
      uint robot_id_;
      float inflation_radius_;
      std::vector<std::string> robot_base_frames_;  // Fully qualified frame names

      // ROS Subscribers, Publishers and Action clients
      rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_subscriber_;
      rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_subscriber_;
      rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr exploration_cmd_subscriber_;
      rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr target_publisher_;
      rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr path_publisher_;
      rclcpp::Publisher<mmpf_explore::msg::ExplorationState>::SharedPtr exploration_state_publisher_;
      rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SharedPtr navigation_client_;

      // ROS TF2
      std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
      std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};

      // Attributes
      rclcpp::TimerBase::SharedPtr timer_main_;
      rclcpp::TimerBase::SharedPtr timer_exploration_state_publisher_;

      // Internal counters / state registers
      int zero_target_detections_count_ = 0;

      // Shared variables
      nav_msgs::msg::OccupancyGrid mapData_, costmapData_;
      nav2_msgs::action::NavigateToPose_Goal robot_goal_;
      mmpf_explore::msg::ExplorationState exploration_state_;
      std::mutex mtx_map; 
      std::mutex mtx_costmap; 
      std::mutex mtx_exploration_state;


      // Constants
      const int MAP_PIXEL_OCCUPIED = 100;
      const int MAP_PIXEL_UNKNOWN = -1;
      const int MAP_PIXEL_FREE = 0;

      const int LARGEST_MAP_DISTANCE = 500000;
      const int CELL_POTENTIAL_STEP = 1;
      const int K_ATTRACT = 1;
      const int ROBOT_INTERFERE_RADIUS = 50;

//       #define RESOLUTION 0.05

// #define ETA_REPLUSIVE 3
// #define DIS_OBTSTACLE 6
// #define DISTANCE_THRES_VALID_OBSTACLE 160
// #define THRESHOLD_TRANSFORM 0.5
// #define ROBOT_INTERFERE_RADIUS 50
// #define LARGEST_MAP_DISTANCE 500000 // 500*1000 
  };
}