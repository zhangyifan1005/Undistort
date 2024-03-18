#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <memory>

class Undistort
{
  public:
      Undistort() = delete;
      //构造函数，参数为：图片宽度，图片高度，相机内参，畸变系数
      Undistort(const int width,const int height,const cv::Mat &intrinsic_matrix,const cv::Mat distortion)
      {
        if(width <= 0 || height <= 0)
        {
          throw std::invalid_argument("Undistort's argument width and height must bigger than 0");
        }

        m_width = width;
        m_height = height;

        //使用opencv的方法获取两个由无畸变坐标到原图坐标映射的map
        cv::initUndistortRectifyMap(intrinsic_matrix, distortion, cv::Mat::eye(3, 3, CV_32F), intrinsic_matrix, cv::Size(m_width, m_height), CV_32FC1, m_mapx, m_mapy);

        //构造一个包含图中所有像素的vector
        std::vector<float> p_data(m_width * m_height * 2);
        int index = 0;
        for (int x = 0; x < m_width; x++)
        {
          for (int y = 0; y < m_height; y++)
          {
            p_data[index++] = static_cast<float>(x);
            p_data[index++] = static_cast<float>(y);
          }
        }

        //由这个vector构造cv::Mat
        cv::Mat points = cv::Mat_<float>(m_width * m_height,2, p_data.data());

        //再使用opencv的多个点去畸变  获取原图到有畸变图片的映射表
        cv::undistortPoints(points, m_undistort_points, intrinsic_matrix, distortion, cv::noArray(), intrinsic_matrix);
      }

      std::pair<int,int> GetUndistortPoint(const int x,const int y)
      {
        if(x <= 0 || x > m_width || y <= 0 || y > m_height)
        {
          throw std::invalid_argument("Undistort GetUndistortPoint's argument width and height error");
        }

        int undistort_x = std::round(m_undistort_points.at<float>(x * m_height + y,0));
        int undistort_y = std::round(m_undistort_points.at<float>(x * m_height + y,1));

        return std::make_pair(undistort_x,undistort_y);
      }

      std::pair<int,int> GetOriginPoint(const int x,const int y)
      {
        if(x <= 0 || x > m_width || y <= 0 || y > m_height)
        {
          throw std::invalid_argument("Undistort GetOriginPoint's argument width and height error");
        }

        //这里需要注意，由于cv::Mat和像素的宽高是反过来表示的，所以通过宽高去取偏移量应该是（y,x）
        int origin_x = std::round(m_mapx.at<float>(y,x));
        int origin_y = std::round(m_mapy.at<float>(y,x));

        return std::make_pair(origin_x,origin_y);
      }
      
  private:
      int m_width;
      int m_height;
      cv::Mat m_mapx;
      cv::Mat m_mapy;
      cv::Mat m_undistort_points;
};

int main(int argc, char **argv)
{
    if(argc > 4)
    {
      //输入参数：原图（畸变） 无畸变图像 原图输出验证 无畸变图像输出验证
      std::string input_ori_path = argv[1];
      std::string input_undistort_path = argv[2];
      std::string output_ori_path = argv[3];
      std::string output_undistort_path = argv[4];


      cv::Mat ori_img = cv::imread(input_ori_path);
      cv::Mat undistort_img =cv::imread(input_undistort_path);

      //读取相机内参及畸变系数
      cv::Mat intrinsic_matrix = (cv::Mat_<float>(3,3) << 2.121681721718232438e+03,0.000000000000000000e+00,1.895012004804071921e+03
                                  ,0.000000000000000000e+00,2.117549237622574310e+03,1.050755447331050391e+03
                                  ,0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00);
      cv::Mat distortion = (cv::Mat_<float>(1,5) << -3.609522335233882051e-02,-2.746143827407116628e-02,3.257971946657706565e-04,1.718441290499933809e-03,0.000000000000000000e+00);

      //将原始标定的点画框
      cv::Point ori_p1(765,567),ori_p2(1056,931);
      cv::Rect ori_rect(ori_p1,ori_p2);

      cv::rectangle(ori_img,ori_rect,cv::Scalar(0,255,0),2);

      //初始化Undistort对象
      std::unique_ptr<Undistort> undistort = std::make_unique<Undistort>(3840,2160,intrinsic_matrix,distortion);

      //获取去畸变后标定点的坐标 并在图上画框
      auto [undistort_x1,undistort_y1] = undistort->GetUndistortPoint(765,567);
      auto [undistort_x2,undistort_y2] = undistort->GetUndistortPoint(1056,931);
      cv::Point undistort_p1(undistort_x1,undistort_y1),undistort_p2(undistort_x2,undistort_y2);
      cv::Rect undistort_rect(undistort_p1,undistort_p2);

      cv::rectangle(ori_img,undistort_rect,cv::Scalar(0,0,255),2);

      cv::imwrite(output_ori_path,ori_img);

      //将两个框分别画在两张图上 看图评估去畸变效果
      cv::rectangle(undistort_img,ori_rect,cv::Scalar(0,255,0),2);
      cv::rectangle(undistort_img,undistort_rect,cv::Scalar (0,0,255),2);
      cv::imwrite(output_undistort_path,undistort_img);

      //控制台看准确坐标
      std::cout << "Original Points: (765,567) , (1056,931)" << std::endl;
      std::cout << "Undistorted Points: (" << undistort_x1 << "," << undistort_y1 << ") , (" << undistort_x2 << "," << undistort_y2 << ")" << std::endl;

      //再将控制台输出的坐标反向验证 看看由无畸变图片的坐标到原图的转换是否准确
      auto [test_x1,test_y1] = undistort->GetOriginPoint(743,557);
      auto [test_x2,test_y2] = undistort->GetOriginPoint(1048,929);
      //查看控制台准确坐标
      std::cout << "tmp Points: (" << test_x1 << "," << test_y1 << ") , (" << test_x2 << "," << test_y2 << ")" << std::endl;
    }

  return 0;
}
