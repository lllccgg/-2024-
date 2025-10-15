#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

Mat img = imread("//home//lin//图片//irving.jpg");

class pub : public rclcpp::Node
{
public:
    pub(std::string name) : Node(name)
    {
        RCLCPP_INFO(this->get_logger(),"我是%s,发布开始。" ,name.c_str());
        pub_str = this->create_publisher<std_msgs::msg::String>("string",10);
        pub_mat = this->create_publisher<sensor_msgs::msg::Image>("mat",10);
        timer = this->create_wall_timer(std::chrono::seconds(5),std::bind(&pub::timer_callback, this));
        
        this->declare_parameter<int>("num",21);
        this->declare_parameter<String>("color","blue");

        // 设置消息头（保持时间戳和坐标系）
        cv_img.header = std_msgs::msg::Header();
        // 设置编码格式
        cv_img.encoding = sensor_msgs::image_encodings::BGR8;
        // 设置图像数据
        cv_img.image = img;
        // 转换为ROS2消息
        pub_img = cv_img.toImageMsg();
        
    }
    void pub_stringAndmat(void)
    {
        // string类型的消息发布
        std::string msgs = "hello";
        std_msgs::msg::String msg;
        msg.data = msgs;
        pub_str->publish(msg); 
        RCLCPP_INFO(this->get_logger(),"string类型的消息已发布。");
        // Mat类型的消息发布
        pub_mat->publish(*pub_img);
        RCLCPP_INFO(this->get_logger(),"Mat类型的消息已发布。");
    }

private:
    // 声明string的发布者
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_str;
    rclcpp::TimerBase::SharedPtr timer;
    // 声明Mat的发布者（先从Mat转化为ROS中的sensor_msgs::msg::Image）
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_mat;
    cv_bridge::CvImage cv_img;
    sensor_msgs::msg::Image::SharedPtr pub_img;
    void timer_callback(void)
    {
        std::string msgs = "hello";
        std_msgs::msg::String msg;
        msg.data = msgs;
        pub_str->publish(msg);

        int num = this->get_parameter("num").as_int();
        String color = this->get_parameter("color").as_string();
        RCLCPP_INFO(this->get_logger(),"%d",num);
        RCLCPP_INFO(this->get_logger(),"%s",color.c_str());
    }
};


int main(int argc,char** argv)
{
    rclcpp::init(argc,argv);
    auto node = std::make_shared<pub>("lcg");
    node->pub_stringAndmat();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

