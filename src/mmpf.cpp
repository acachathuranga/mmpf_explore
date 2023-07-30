/* Copyright (c) 2020 Georgia Instititue of Technology 
 * Copyright (c) 2020 Tsinghua University
 * Copyright (c) 2023 Singapore University of Technology
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
 * Author: Achala Athukorala (chhathuranga@gmail.com)
 */

#include "mmpf_explore/mmpf.hpp"

using namespace exploration;
using std::placeholders::_1;

MMPF::MMPF() : Node("MMPF_node")
{
    /*------- Fetch parameters ------*/
    if (!get_ros_parameters()) return; // Exit if parameters fetching failure

    /*------- Initialize Exploration state ------*/
    exploration_state_.robot_id = robot_id_;
    exploration_state_.status = mmpf_explore::msg::ExplorationState::INACTIVE;

    /*------- Create navigation_stack action client ------*/
    navigation_client_ = rclcpp_action::create_client<nav2_msgs::action::NavigateToPose>(this, "navigate_to_pose");

    /*------- Initialize TF listener ------*/
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    /*------- Create Subscribers and publishers ------*/
    map_subscriber_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(map_topic_, 10, std::bind(&MMPF::map_callback, this, _1));
    costmap_subscriber_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(costmap_topic_, 20, std::bind(&MMPF::costmap_callback, this, _1));
    exploration_cmd_subscriber_ = this->create_subscription<std_msgs::msg::Bool>(std::string(this->get_name()) + "/enable", 10, std::bind(&MMPF::enable_exploration_callback, this, _1));
    target_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("frontier_targets", 1);
    path_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("frontier_path", 1);
    exploration_state_publisher_ = this->create_publisher<mmpf_explore::msg::ExplorationState>("/exploration_state", 1);
    
    /*------- Create timers ------*/
    // timer_exploration_state_publisher_ = this->create_wall_timer( std::chrono::duration<double>(1.0 / rate_), std::bind(&RRT::publish_exploration_state, this));

}

void MMPF::explore()
{
    /*------- Return if maps are not available ------*/
    if (!map_data_available()) return;

    /*------- Return if self map to base_foot_print transform is not available ------*/
    geometry_msgs::msg::TransformStamped map_to_baseframe;
    if (!get_transform(map_frame_, robot_base_frame_, map_to_baseframe)) return;

    /*------- Exploration state thread safety lock ---------*/
    std::unique_lock<std::mutex> exploration_state_lock(mtx_exploration_state);

    /*------- Fetch external data------*/
    nav_msgs::msg::OccupancyGrid mapData = get_map_data();  
    nav_msgs::msg::OccupancyGrid costmapData = get_costmap_data(); 

    /*------- Find Frontiers ------*/
    std::vector<Pixel> targets;
    find_frontiers(mapData, costmapData, targets);

    /*------- Exploration Completion Detection ------*/
    if (targets.size() == 0) {
        zero_target_detections_count_++;

        if (zero_target_detections_count_ >= 5) {
            RCLCPP_INFO_STREAM(this->get_logger(), "Exploration Done!");
            this->navigation_client_->async_cancel_all_goals();
            this->timer_main_->cancel();
            exploration_state_.status = mmpf_explore::msg::ExplorationState::DONE;
            zero_target_detections_count_ = 0;  // Reset counter
        }
        return;
    } else {
        zero_target_detections_count_ = 0;  // Reset counter
    }

    /*------- Cluster targets into different groups and find the center of each group   ------*/
    std::vector<Cluster> target_clusters = cluster_2D(targets);

    /*------- Display cluster centroids   ------*/
    visualization_msgs::msg::Marker target_cluster_markers = create_visualization_msg(VisualizationType::POINTS);
    for (auto &cluster : target_clusters)
    {
        target_cluster_markers.points.push_back(geometry_msgs::msg::Point().set__x(
                    cluster.x * mapData.info.resolution + mapData.info.origin.position.x).set__y(
                    cluster.y * mapData.info.resolution + mapData.info.origin.position.y));
    }
    target_publisher_->publish(target_cluster_markers);

    /*------- Generate distance map for each target (potential maps) ------*/
    std::vector<std::vector<int>> target_potential_maps;
    target_potential_maps.reserve(target_clusters.size());
    for (auto &cluster : target_clusters)
    {
        std::vector<int> distance_map;
        create_potential_map(mapData, cluster.center(), distance_map, CELL_POTENTIAL_STEP);
        target_potential_maps.push_back(distance_map);
    }

    /*------- Receive other robots' locations ------*/
    std::vector<geometry_msgs::msg::Transform> peer_robot_transforms;
    for (int i = 0; i < robot_count_; i++) 
    {
        if ((uint)(i + 1) == robot_id_) continue;   // Skip hostrobot
        geometry_msgs::msg::TransformStamped transform;
        if (get_transform(map_frame_, robot_base_frames_[i], transform)) {
            peer_robot_transforms.push_back(transform.transform);
        }
    }

    /*------- Calculate path ------*/
    Pixel robot_location((map_to_baseframe.transform.translation.x - mapData.info.origin.position.x) / mapData.info.resolution,
                         (map_to_baseframe.transform.translation.y - mapData.info.origin.position.y) / mapData.info.resolution);
    std::vector<Pixel> path = calculate_exploration_path(robot_location, 
                                                            target_clusters, 
                                                            target_potential_maps, 
                                                            peer_robot_transforms, 
                                                            mapData.info);

    /*------- Send Navigation Goal ------*/
    robot_goal_.pose.header.stamp = rclcpp::Time(0);
    robot_goal_.pose.pose.position.x = path.back().x * mapData.info.resolution + mapData.info.origin.position.x;
    robot_goal_.pose.pose.position.y = path.back().y * mapData.info.resolution + mapData.info.origin.position.y;
    RCLCPP_INFO_STREAM(this->get_logger(), "Goal: " << robot_goal_.pose.pose.position.x << ", " << robot_goal_.pose.pose.position.y);
    this->navigation_client_->async_send_goal(robot_goal_);

    return;
}

/**
 * @brief Extracts obstacle points and exploration targets given a robot map and costmap
 * 
 * @param map 
 * @param costmap 
 * @param obstacles 
 * @param targets 
 */
void MMPF::find_frontiers(nav_msgs::msg::OccupancyGrid mapData, nav_msgs::msg::OccupancyGrid costmapData, std::vector<Pixel> &targets)
{
     /*------- Initialize the map ------*/
    int map_height = mapData.info.height;
    int map_width = mapData.info.width;
    std::vector<int> map(mapData.data.begin(), mapData.data.end());
    std::vector<Pixel> obstacles;
    std::list<Pixel> target_list;
    
    /*-------  Find targets & Obstacles ------*/
    // Reserve max sizes for vectors to prevent relocation
    obstacles.reserve(map_height * map_width);

    // Traverse map row, column wise while checking each pixel for free regions
    for (int i = 2; i < (map_height - 2); i++)
    {
        for (int j = 2; j < (map_width - 2); j++)
        {
            if (map[i*map_width + j] == MAP_PIXEL_OCCUPIED)
            {
                obstacles.emplace_back(j,i);
            }
            else if (map[i*map_width + j] == MAP_PIXEL_UNKNOWN)
            {
                // accessiable frontiers
                int numFree = 0, temp1 = 0;

                if (map[(i + 1)*map_width + j] == 0){
                    temp1 += (map[(i + 2)*map_width + j    ] == 0) ? 1 : 0;
                    temp1 += (map[(i + 1)*map_width + j + 1] == 0) ? 1 : 0;
                    temp1 += (map[(i + 1)*map_width + j - 1] == 0) ? 1 : 0;
                    numFree += (temp1 > 0);
                }

                if (map[i*map_width + j + 1] == 0){
                    temp1 = 0;
                    temp1 += (map[      i*map_width + j + 2] == 0) ? 1 : 0;
                    temp1 += (map[(i + 1)*map_width + j + 1] == 0) ? 1 : 0;
                    temp1 += (map[(i - 1)*map_width + j + 1] == 0) ? 1 : 0;
                    numFree += (temp1 > 0);
                }

                if (map[(i - 1) *map_width + j] == 0){
                    temp1 = 0;
                    temp1 += (map[(i - 1)*map_width + j + 1] == 0) ? 1 : 0;
                    temp1 += (map[(i - 1)*map_width + j - 1] == 0) ? 1 : 0;
                    temp1 += (map[(i - 2)*map_width + j    ] == 0) ? 1 : 0;
                    numFree += (temp1 > 0);
                }

                if (map[i * map_width + j - 1] == 0){
                    temp1 = 0;
                    temp1 += (map[    i  *map_width + j - 2] == 0) ? 1 : 0;
                    temp1 += (map[(i + 1)*map_width + j - 1] == 0) ? 1 : 0;
                    temp1 += (map[(i - 1)*map_width + j - 1] == 0) ? 1 : 0;
                    numFree += (temp1 > 0);
                }

                if( numFree > 0 ) {
                    target_list.emplace_back(j,i);
                }
            }
        }
    }
    // Shrink obstacle vector
    obstacles.shrink_to_fit();

    // TODO Border traversal is not included in original RRT Algorithm. Hence commented
    // // Border traversal (Check if any free cell lies on map borders)
    // for (int i = 0; i < map_height; i++)
    // {
    //     if (map[i * map_width] == MAP_PIXEL_FREE) {                     // Left border
    //         targets.emplace_back(0, i);
    //     }                   
    //     if (map[map_width - 1 + i * map_width] == MAP_PIXEL_FREE) {     // Right border
    //         targets.emplace_back(map_width-1, i);            
    //     }
    // }

    // for (int j = 0; j < map_width; j++)
    // {
    //     if (map[j] == MAP_PIXEL_FREE) {                                 // Bottom border
    //         targets.emplace_back(j, 0);
    //     }                   
    //     if (map[(map_height - 1) * map_width + j] == MAP_PIXEL_FREE) {  // Top border
    //         targets.emplace_back(j, map_height - 1);            
    //     }
    // }

    /*-------  Remove targets within the inflation layer of costmap ------*/
    for (auto target=target_list.begin(); target!=target_list.end();)
    {
        float loc_x = target->x * mapData.info.resolution + mapData.info.origin.position.x;
        float loc_y = target->y * mapData.info.resolution + mapData.info.origin.position.y;
        int index_costmap = (loc_y - costmapData.info.origin.position.y) / costmapData.info.resolution * costmapData.info.width +
                            (loc_x - costmapData.info.origin.position.x)/costmapData.info.resolution;
        
        if (costmapData.data[index_costmap] > MAP_PIXEL_FREE) {
            target = target_list.erase(target);
        } else {
            target++;
        }
    }

    /*-------  Remove targets within inflation radius of obstacles ------*/
    for (auto target=target_list.begin(); target!=target_list.end();)
    {
        for (auto obstacle : obstacles) {
            if (std::abs(target->y - obstacle.y) + std::abs(target->x - obstacle.x) < inflation_radius_) {
                target = target_list.erase(target);
                break;
            }
        }
        target++;
    }

    // Copy targets to vector
    targets.resize(target_list.size());
    std::copy(target_list.begin(), target_list.end(), targets.begin());
    return;
}

/**
 * @brief Given a set of 2D points, returns a set of clusters
 * 
 * @param points 
 * @param proximity_threshold (optional) Maximum distance between two points, for them to be considered in one cluster (default = 3)
 * @return std::vector<MWFCN::Cluster> A vector of Clusters : [cluster center(x,y), number of points within the cluster]
 */
std::vector<MMPF::Cluster> MMPF::cluster_2D(std::vector<MMPF::Pixel> points, int proximity_threshold)
{
    // Note: x & y value of detected targets are in increasing order because of the detection is in laser scan order.
    std::vector<Cluster> clusters;
    clusters.reserve(points.size());

    std::list<Pixel> point_list(points.begin(), points.end());
    std::list<std::list<Pixel>> point_clusters; // List of point clusters (A single point cluster is a list of pixel points)

    do
    {
        /*-------  Create a new cluster ------*/
        point_clusters.emplace_back();
        point_clusters.back().push_back(point_list.back());
        double center_x = point_list.back().x;
        double center_y = point_list.back().y;
        point_list.pop_back();
        

        /*------- Traverse current cluster while checking points in close proximity  ------*/
        for (auto clustered_point=point_clusters.back().begin(); clustered_point!=point_clusters.back().end(); clustered_point++)
        {
            for (auto unclustered_point=point_list.begin(); unclustered_point!=point_list.end();)
            {
                if ((abs(clustered_point->x - unclustered_point->x) + abs(clustered_point->y - unclustered_point->y)) < proximity_threshold)
                {
                    // Found unclustered point closer to current cluster. Move point to cluster
                    point_clusters.back().emplace_back(*unclustered_point);
                    // Add to cluster center value total
                    center_x += unclustered_point->x;
                    center_y += unclustered_point->y;

                    // Erase point from unclustered list and get iterator to next unclustered point
                    unclustered_point = point_list.erase(unclustered_point);
                }
                else
                {
                    unclustered_point++;
                }
            }
        }

        /*-------  Calculate cluster center and find data point closest to center ------*/
        center_x /= point_clusters.back().size();
        center_y /= point_clusters.back().size();

        /*------- Find closest data point to cluster center  ------*/
        float min_dist = 100.0;
        Pixel center_point(0,0);
        for (auto &point : point_clusters.back())
        {
            if( (abs(point.x - center_x) + abs(point.y - center_y) ) < min_dist)
            {
                min_dist = abs(point.x - center_x) + abs(point.y - center_y);
                center_point = point;
            }
        }

        /*------- Add cluster data to cluster vector  ------*/
        clusters.emplace_back(center_point, point_clusters.back().size());

    } while (!point_list.empty());
    
    return clusters;
}

/**
 * @brief For a given OccupancyGrid, creates a potential map (distance map) taking SourcePoint as starting location (distance = 0)
 * 
 * @param mapData           OccupancyGrid map data
 * @param source_point      Starting point (Zero potential point)
 * @param potential_map
 * @param potential_step    (optional) Potential / Distance increase between two adjacent points in the map. (default 3)      
 * @return true     if potential map creation is successfull
 * @return false    if potential map creation failed
 */
bool MMPF::create_potential_map(nav_msgs::msg::OccupancyGrid mapData, Pixel source_point, std::vector<int> &potential_map, int potential_step)
{
    /*------- Initialize potential map with maximum potential ------*/
    potential_map.resize(mapData.data.size());
    potential_map.assign(potential_map.size(), LARGEST_MAP_DISTANCE);

    /*------- Initialize map and current processing points ------*/
    std::vector<int> map(mapData.data.begin(), mapData.data.end());
    int width = mapData.info.width;
    std::vector<Pixel> process_queue;
    process_queue.emplace_back(source_point);
    potential_map[source_point.y * width + source_point.x] = 0;

    int iteration = 1;
    while (process_queue.size() > 0)
    {
        /*
            For all pixels in process_queue, check if adjacent pixels are free.
            Add each adjacent free pixel to adjacent_pixels. Update potentials for added pixels in potential map.
            When adding pixels to adjacent_pixels, set pixel to occupied, to prevent re-processing

            After processing all points in process_queue, replace process_queue with points in adjacent_pixels and repeat process
        */
        std::vector<Pixel> adjacent_pixels;
        for (auto &point : process_queue)
        {
            // Check right pixel
            process_pixel_potential(point, Pixel(point.x + 1, point.y), map, potential_map, width, adjacent_pixels, potential_step);
            // Check left pixel
            process_pixel_potential(point, Pixel(point.x - 1, point.y), map, potential_map, width, adjacent_pixels, potential_step);
            // Check up pixel
            process_pixel_potential(point, Pixel(point.x, point.y + 1), map, potential_map, width, adjacent_pixels, potential_step);
            // Check down pixel
            process_pixel_potential(point, Pixel(point.x, point.y - 1), map, potential_map, width, adjacent_pixels, potential_step);
        }
        process_queue = adjacent_pixels;
        adjacent_pixels.clear();
        iteration++;

        // Costmap expansion maximum iteration limit check  ( TODO : Check if this is necessary)
        if (iteration > LARGEST_MAP_DISTANCE)
        {
            RCLCPP_WARN_STREAM(this->get_logger(), "Distance exceeds MAXIMUM SETUP");
            return false;
        }
    }
    return true;
}

/**
 * @brief Given a source pixel and a target pixel, checks if the target pixel is free, calculates its cost and adds to discovered pixels
 * 
 * @param source_pixel          A pixel with a known cost
 * @param target_pixel          An adjacent pixel to source pixel
 * @param map                   A vector containing map data
 * @param potential_map         A vector containing costs for pixels (including source pixel cost)
 * @param map_width             Width of map or potential_map
 * @param discovered_pixels     A vector of free pixels around source_pixel
 * @param unit_potential        Potential increase between adjacent pixels
 */
inline void MMPF::process_pixel_potential(Pixel source_pixel, 
                                            Pixel target_pixel, 
                                            std::vector<int> &map, 
                                            std::vector<int> &potential_map, 
                                            int map_width, 
                                            std::vector<Pixel> &discovered_pixels,
                                            int unit_potential)
{
    // Check map bounds
    if ( ((target_pixel.x + target_pixel.y * map_width) < 0) || ((target_pixel.x + target_pixel.y * map_width) >= (int)map.size()) )
    {
        return;
    }

    // Potential map is extended to unknown regions also
    if ( (map[target_pixel.x + target_pixel.y * map_width] == MAP_PIXEL_FREE) || (map[target_pixel.x + target_pixel.y * map_width] == MAP_PIXEL_UNKNOWN) )
    {
        discovered_pixels.emplace_back(target_pixel.x, target_pixel.y);
        potential_map[target_pixel.x + target_pixel.y * map_width] = potential_map[source_pixel.x + source_pixel.y * map_width] + unit_potential;
        map[target_pixel.x + target_pixel.y * map_width] = MAP_PIXEL_OCCUPIED;
    }
    return;
}

/**
 * @brief Calculates a minimum potential path to a frontier cluster for exploration
 * 
 * @param current_location          Host Robot location
 * @param target_clusters           Frontier Clusters
 * @param cluster_potential_maps    Potential map (Distance map) starting from each cluster center
 * @param peer_robot_transforms     Other robots' transforms
 * @param map_info                  Map information (origin, resolution, width, height .etc)
 * @return std::vector<MMPF::Pixel> 
 */
std::vector<MMPF::Pixel> MMPF::calculate_exploration_path(MMPF::Pixel current_location,
                                                            std::vector<MMPF::Cluster> target_clusters,
                                                            std::vector<std::vector<int>> &cluster_potential_maps,
                                                            std::vector<geometry_msgs::msg::Transform> peer_robot_transforms,
                                                            nav_msgs::msg::MapMetaData map_info)
{
    std::vector<MMPF::Pixel> path;
    path.push_back(current_location);
    visualization_msgs::msg::Marker path_msg = create_visualization_msg(VisualizationType::LINE);

    int iteration = 1;
    int currentPotential = 10000;
    int riverFlowPotentialGain = 1;
    float minDis2Frontier  = 10000;  // a random initialized value greater than all possible distances.
    while(iteration < 3000 && minDis2Frontier > 1)
    {
        /*------- Get the minimal potential of the points around the current location ------*/
        float potential[4];
        int min_idx = -1;
        float min_potential = 10000;
        Pixel loc_around[4]; // Surrounding pixels at current location

        loc_around[0] = Pixel(current_location.x    , current_location.y + 1);  // Up
        loc_around[1] = Pixel(current_location.x - 1, current_location.y + 1);  // Left
        loc_around[2] = Pixel(current_location.x    , current_location.y - 1);  // Down
        loc_around[3] = Pixel(current_location.x + 1, current_location.y    );  // Right

        // Calculate potentials of neighbourhood
        for (int i = 0; i < 4; i++)
        {   
            // Calculate potential of considered neighbor
            Pixel curr_around = loc_around[i];
            float attract = 0;
            for (uint j = 0; j < target_clusters.size(); j++) {
                float temp = float(cluster_potential_maps[j][curr_around.x + curr_around.y * map_info.width]);
                if (temp < 1) continue;
                attract = attract - K_ATTRACT * target_clusters[j].size / temp;
            }

            // Increase potential, if current point has been passed before
            for (auto &point : path) {
                if (curr_around == point) {
                    attract += riverFlowPotentialGain * 5;
                }
            }

            // Add impact of peer robots
            for (auto &robot_tf : peer_robot_transforms){
                Pixel robot_location((robot_tf.translation.x - map_info.origin.position.x) / map_info.resolution, 
                                     (robot_tf.translation.y - map_info.origin.position.y) / map_info.resolution);
                int dis = abs(curr_around.x - robot_location.x) + abs(curr_around.y - robot_location.y);
                if (dis < ROBOT_INTERFERE_RADIUS) {
                    int temp = (ROBOT_INTERFERE_RADIUS - dis);
                    attract += temp;
                }
            }

            potential[i] = attract;
            if (min_potential > potential[i]){
                min_potential = potential[i];
                min_idx = i;
            }
        }
        
        if (currentPotential > min_potential){
            path.push_back(loc_around[min_idx]);
            currentPotential = min_potential;
        } else {
            riverFlowPotentialGain++;
        }

        current_location = path.back();

        for (uint i = 0; i < target_clusters.size(); i++){
            int temp_dis = cluster_potential_maps[i][current_location.x + current_location.y * map_info.width];
            if ((temp_dis == 0) && 
                ((abs(current_location.x - target_clusters[i].x) + abs(current_location.y - target_clusters[i].y)) > 0)) {
                continue;
            }

            if (minDis2Frontier > temp_dis){
                minDis2Frontier = temp_dis;
            }
        }
        iteration++;

        /*------- Display path   ------*/
        if (path.size() >= 2) {
            path_msg.points.push_back(geometry_msgs::msg::Point().set__x(
                        path[path.size() - 2].x * map_info.resolution + map_info.origin.position.x).set__y(
                        path[path.size() - 2].y * map_info.resolution + map_info.origin.position.y));

            path_msg.points.push_back(geometry_msgs::msg::Point().set__x(
                        current_location.x * map_info.resolution + map_info.origin.position.x).set__y(
                        current_location.y * map_info.resolution + map_info.origin.position.y));
            path_publisher_->publish(path_msg);
        }
    }
    return path;
}

/**
 * @brief 
 * 
 * @return true     If success
 * @return false    If failure
 */
bool MMPF::get_ros_parameters()
{
    this->declare_parameter("map_topic", "map"); 
    this->declare_parameter("costmap_topic", "global_costmap/costmap");
    this->declare_parameter("robot_base_frame", "base_footprint");
    this->declare_parameter("robot_frame_prefix", "robot");
    this->declare_parameter("rate", 1.0);
    this->declare_parameter("robot_count", 1);
    this->declare_parameter("inflation_radius", 6.0);
    
    
    map_topic_ = this->get_parameter("map_topic").get_parameter_value().get<std::string>();
    costmap_topic_ = this->get_parameter("costmap_topic").get_parameter_value().get<std::string>();
    robot_base_frame_ = this->get_parameter("robot_base_frame").get_parameter_value().get<std::string>();
    robot_frame_prefix_ = this->get_parameter("robot_frame_prefix").get_parameter_value().get<std::string>();
    rate_ = this->get_parameter("rate").get_parameter_value().get<float>();
    robot_count_ = this->get_parameter("robot_count").get_parameter_value().get<int>();
    inflation_radius_ = this->get_parameter("inflation_radius").get_parameter_value().get<float>();
    
    // Remove any leading slash from robot_base_frame
    if (*robot_base_frame_.cbegin() == '/') robot_base_frame_.erase(0, 1);
    // Create fully qualified robot_base_frame names
    for (int i = 1; i < robot_count_ + 1; i++)
    {
        robot_base_frames_.push_back(robot_frame_prefix_ + std::to_string(i) + "/" + robot_base_frame_);
    }

    // Extract robot id from node namespace
    std::string ns = this->get_namespace();
    RCLCPP_INFO_STREAM(this->get_logger(), "Running exploration in namespace: " << ns);
    try{
        std::string id_string = ns.substr(robot_frame_prefix_.size() + 1);
        robot_id_ = std::stoi(id_string);
        RCLCPP_INFO_STREAM(this->get_logger(), "Robot ID: " << robot_id_);
    }
    catch( ... ){
        RCLCPP_ERROR_STREAM(this->get_logger(), "Cannot extract robot id from node namespace {" << ns 
                                                    << "} with prefix {" << robot_frame_prefix_ << "}");
        return false;
    }

    #ifdef DEBUG
        RCLCPP_INFO_STREAM(MMPF::get_logger(), "map topic: " << map_topic_
        <<"\ncostmap_topic: " << costmap_topic_
        <<"\nrobot_base_frame: " << robot_base_frame_
        <<"\nrate: " << rate_
        <<"\nrobot_count: " << robot_count_
        <<"\nrobot_frame_prefix: " << robot_frame_prefix_
        <<"\ninflation_radius: " << inflation_radius_
        );
    #endif

    return true;
}

/**
 * @brief Given two frames, finds the transform from source to target frame
 * 
 * @param target_frame 
 * @param source_frame 
 * @param transform 
 * @return true     If transform is available
 * @return false    If transform is not available
 */
bool MMPF::get_transform(std::string target_frame, std::string source_frame, geometry_msgs::msg::TransformStamped &transform)
{
    try{
        transform = tf_buffer_->lookupTransform(target_frame, source_frame,tf2::TimePointZero);
        return true;
    }
    catch( const tf2::TransformException & ex){
        RCLCPP_WARN_STREAM(this->get_logger(), source_frame + " to " + target_frame + " transform is not available : " + ex.what());
    }
    return false;
}

void MMPF::map_callback(const nav_msgs::msg::OccupancyGrid msg)
{
    set_map_data(msg);
}

void MMPF::costmap_callback(const nav_msgs::msg::OccupancyGrid msg)
{
    set_costmap_data(msg);
}

void MMPF::enable_exploration_callback(const std_msgs::msg::Bool msg)
{
    if (msg.data)
    {
        if (exploration_state_.status != mmpf_explore::msg::ExplorationState::ACTIVE)
        {
            /*------- Initialize Exploration state ------*/
            exploration_state_.status = mmpf_explore::msg::ExplorationState::ACTIVE;
            
            RCLCPP_INFO_STREAM(this->get_logger(), "Starting Exploration!");

            /*------- Create main callback timer ------*/
            timer_main_ = this->create_wall_timer( std::chrono::duration<double>( 1.0 / rate_ ), std::bind(&MMPF::explore, this));
        }
        else
        {
            RCLCPP_WARN_STREAM(this->get_logger(), "Exploration already running. Start request ignored!");
        }
    }
    else
    {
        this->timer_main_->cancel();
        this->navigation_client_->async_cancel_all_goals(); 
        RCLCPP_INFO_STREAM(this->get_logger(), "Ending exploration: Requested by User!");
        exploration_state_.status = mmpf_explore::msg::ExplorationState::DONE;
    }
}

void MMPF::set_costmap_data(nav_msgs::msg::OccupancyGrid costmapData)
{
    std::unique_lock<std::mutex> lck (mtx_costmap);
    costmapData_=costmapData;
}

nav_msgs::msg::OccupancyGrid MMPF::get_costmap_data()
{
    std::unique_lock<std::mutex> lck (mtx_costmap);
    return costmapData_;
}

void MMPF::set_map_data(nav_msgs::msg::OccupancyGrid mapData)
{
    std::unique_lock<std::mutex> lck (mtx_map);
    mapData_=mapData;
}

nav_msgs::msg::OccupancyGrid MMPF::get_map_data()
{
    std::unique_lock<std::mutex> lck (mtx_map);
    return mapData_;
}

bool MMPF::map_data_available(){

    if (!(get_map_data().data.size() < 1)){
        map_frame_ = get_map_data().header.frame_id;
        robot_goal_.pose.header.frame_id = map_frame_;
        robot_goal_.pose.pose.position.z = 0;
        robot_goal_.pose.pose.orientation.z = 1.0;
        return true;
    }

    RCLCPP_WARN_STREAM(this->get_logger(), "map data is not available");
    return false;
}  

void MMPF::publish_exploration_state(void)
{
    std::unique_lock<std::mutex> exploration_state_lock(mtx_exploration_state);

    /*------- Update Exploration state ------*/
    geometry_msgs::msg::TransformStamped map_to_baseframe;
    if (!get_transform(map_frame_, robot_base_frame_, map_to_baseframe)) return;
    exploration_state_.header.frame_id = robot_frame_prefix_ + std::to_string(robot_id_) + "/" + map_frame_;
    exploration_state_.location.position.set__x(map_to_baseframe.transform.translation.x). set__y(
        map_to_baseframe.transform.translation.y).set__z(map_to_baseframe.transform.translation.z);
    exploration_state_.location.orientation.set__x(map_to_baseframe.transform.rotation.x).set__y(
        map_to_baseframe.transform.rotation.y).set__z(map_to_baseframe.transform.rotation.z).set__w(
        map_to_baseframe.transform.rotation.w);

    exploration_state_.header.stamp = this->get_clock()->now();
    exploration_state_publisher_->publish(exploration_state_);
}

visualization_msgs::msg::Marker MMPF::create_visualization_msg(int type, double lifetime){

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = map_frame_;
    marker.header.stamp = rclcpp::Time(0);
    marker.lifetime         = rclcpp::Duration::from_seconds(lifetime);

    if (type == LINE) {
        //------------------------------------- initilize the visualized lines
        marker.type				= marker.LINE_LIST;
        marker.action           = marker.ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x 			= 0.03;
        marker.scale.y			= 0.03;
        marker.color.r			= 1.0;   // 0.0/255.0;
        marker.color.g			= 0.0;   // 0.0/255.0;
        marker.color.b 			= 1.0;   // 236.0/255.0;
        marker.color.a 			= 1.0;
    }

    else if(type == POINTS) {
        //------------------------------------- initilize the visualized points
        marker.type 			= marker.POINTS;
        marker.action           = marker.ADD;
        marker.pose.orientation.w =1.0;
        marker.scale.x 			= 0.3; 
        marker.scale.y			= 0.3; 
        marker.color.r 			= 1.0;   // 255.0/255.0;
        marker.color.g 			= 0.0;   // 0.0/255.0;
        marker.color.b 			= 0.0;   // 0.0/255.0;
        marker.color.a			= 1.0;
    }

    else if(type == SPHERES) {
        //------------------------------------- initilize the visualized points
        marker.type 			= marker.SPHERE_LIST;
        marker.action           = marker.ADD;
        marker.pose.orientation.w =1.0;
        marker.scale.x 			= 0.3; 
        marker.scale.y			= 0.3; 
        marker.color.r 			= 1.0;   // 255.0/255.0;
        marker.color.g 			= 0.0;   // 0.0/255.0;
        marker.color.b 			= 0.0;   // 0.0/255.0;
        marker.color.a			= 1.0;
    }

    else{
        RCLCPP_ERROR_STREAM(MMPF::get_logger(), "Undefined visualization msg type");
    }
    return marker;
 } 