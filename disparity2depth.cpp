#include <iostream>
​
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
​
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/gpu/kinfu/kinfu.h>
using namespace std;
using namespace cv;
using namespace pcl;
​
class process_images{
public:
  process_images(){
​
    winSize = 3;
    minDisp = 0;
    maxDisp = 208-32;
//    base_line = -0.104886431f;//-0.108715; // cam.proj(1,4)/(1,1)
//    f_norm = 1166.445751;//1507.6098/2;
//    Cx = 1094.387375;//1280.7518/2;
//    Cy = 764.673874;//1022.667/2;
​
    // from field calib 5_27_2020
    base_line = -0.104895623491f;//-0.104886431f;//-0.108715; // cam.proj(1,4)/(1,1)
    f_norm = 1124.423722890519;//1507.6098/2;
    Cx = 1009.989379882812;//1280.7518/2;
    Cy = 772.01904296875;//1022.667/2;
  }
​
  void display_pcl(PointCloud<PointXYZRGB>::Ptr c, std::string name){
      pcl::visualization::PCLVisualizer viewer(name.c_str());
      viewer.addPointCloud<pcl::PointXYZRGB>(c,name.c_str());
      while (!viewer.wasStopped())
          viewer.spinOnce();
  }
​
  void get_disparity(){
​
    l =  imread("/home/agvbotics/Desktop/data/05_27_2020_NY_Test/extracted/bag_7/COLOR/cam0/C0_F000000.png",1);
    r =  imread("/home/agvbotics/Desktop/data/05_27_2020_NY_Test/extracted/bag_7/COLOR/cam1/C1_F000000.png",1);
​
    cout<<"left.size: "<<l.cols<<"x"<<l.rows<<"x"<<l.channels()<<endl;
    cout<<"right.size: "<<r.cols<<"x"<<r.rows<<"x"<<r.channels()<<endl;
​
    Mat disp8(l.rows, l.cols,CV_32F);
​
    Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(minDisp,maxDisp,winSize);
    sgbm->setP1(8*winSize*winSize);
    sgbm->setP2(32*winSize*winSize);
    sgbm->setMinDisparity(minDisp);
    sgbm->setNumDisparities(maxDisp);
    sgbm->setUniquenessRatio(60); //50 // higher the better but less details
    sgbm->setPreFilterCap(50);
    sgbm->setSpeckleWindowSize(30);//50 // higher = more filter = less smaller blobs
    sgbm->setSpeckleRange(1);   //7  lower the better
    sgbm->setDisp12MaxDiff(maxDisp+32);
    sgbm->setMode(StereoSGBM::MODE_HH4); //MODE_SGBM_3WAY
​
    sgbm->compute(l,r,sgbm_disp);
​
    sgbm_disp.convertTo(disp8, CV_32F, 1.0/16.0);
​
    cv::namedWindow("disparity",0);
    cv::imshow("disparity",sgbm_disp);
    cv::waitKey(0);
​
    get_point_cloud(disp8);
​
  }
​
  void get_point_cloud(cv::Mat disp8){
    pcl::PointXYZRGB cloud_xyz;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr CLOUD (new pcl::PointCloud<pcl::PointXYZRGB>);
​
    std::vector<Point3f> coor;
​
    float X, Y, Z;
​
    for(int i=0;i<l.rows;i++){
        for(int j=0;j<l.cols;j++){
​
            Z = (float)((-1* f_norm * base_line) / (disp8.at<float>(i,j)));
            X = (float)(Z/f_norm) * (j - Cx); // cols
            Y = (float)(Z/f_norm) * (i - Cy); // rows
​
            cloud_xyz.x = X;
            cloud_xyz.y = Y;
            cloud_xyz.z = Z;
​
            if ((Z >= 0.5 && Z<1.2)
​
                         //&& ((chs[0].at<uchar>(i,j)<50) && (chs[0].at<uchar>(i,j)>=0))   // brown for all vines
                         //&& ((chs_gan[2].at<uchar>(i,j)>=100) && (chs_gan[2].at<uchar>(i,j)<=255)) //RED is for CANES
                         //&& ((BW.at<uchar>(i,j)>0)) //RED is for CANES
                         //&& ((chs_gan[0].at<uchar>(i,j)>=0)   && (chs_gan[0].at<uchar>(i,j)<=50)) //BLUE
                         //&& ((chs_gan[1].at<uchar>(i,j)>=0)   && (chs_gan[1].at<uchar>(i,j)<=50)) //GREEN
                    ){
                cloud_xyz.r = 255;
                cloud_xyz.g = 255;
                cloud_xyz.b = 255;
​
                CLOUD->points.push_back(cloud_xyz);  // Cane Cloud
                //counter_cane_pix++;
            }
        }
    }
​
    cout<<"CLOUD size: "<<CLOUD->points.size()<<endl;
    //cout<<"pix Counter (cane/ cordon): "<<counter_cane_pix<<"/ "<<counter_cordon_pix<<endl;
    //================================= VOXEL FILTERING =======================================
​
    //============== display PCL ====================
​
    display_pcl(CLOUD,"CLOUD");
​
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr no_nan_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    vector<int> no_nan_index;
    pcl::removeNaNFromPointCloud(*CLOUD,*no_nan_cloud,no_nan_index);
​
    pcl::PointIndices::Ptr removed_indices (new pcl::PointIndices);//, removed_indices_cane (new pcl::PointIndices);
​
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr VF_CLOUD(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr IN_CLOUD(new pcl::PointCloud<pcl::PointXYZRGB>);
    *IN_CLOUD = *no_nan_cloud;
​
    //=========================== VOX FILTERING ===============================================
​
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud(IN_CLOUD);
    sor.setLeafSize (0.005f, 0.005f, 0.005f); // 0.01 0.01 0.03
    sor.filter (*VF_CLOUD);
​
    cout<<"VF_CLOUD size: "<<VF_CLOUD->points.size()<<endl;
​
    //=========================== SOR FILTERING ===============================================
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyz_cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB> ());
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> filter (true);
    filter.setInputCloud (VF_CLOUD);
    filter.setMeanK (10); //5,10
    filter.setStddevMulThresh (0.95); //0.9  0.01  // low STD = more filtering
    filter.setNegative (0);
    filter.setKeepOrganized (false);
    filter.filter (*xyz_cloud_filtered);
    filter.getRemovedIndices (*removed_indices);
​
    cout<<"filtered CLOUD size: "<<xyz_cloud_filtered->points.size()<<endl;
    display_pcl(xyz_cloud_filtered,"xyz_cloud_filtered");
  }
​
protected:
  cv::Mat l,r;
  float base_line;
  int winSize ;
  int minDisp ;
  int maxDisp ; // 208 224 240 256 272 304 400
​
  Mat sgbm_disp;
​
  double f_norm;// =  CamLProj.at<double> (0,0);
  double Cx;// = CamLMatrix.at<double>(0,2);
  double Cy;// = CamLMatrix.at<double>(1,2);
​
};
​
int main(){
​
    process_images p;
    p.get_disparity();
​
    return 0;
}