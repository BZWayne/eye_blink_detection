#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/face.hpp"
#include <opencv2/core/mat.hpp>
#include <stdio.h>
#include <math.h>

#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::face;

void detectFaceEyesAndDisplay( Mat frame );
Point middlePoint(Point p1, Point p2);
float blinkingRatio (vector<Point2f> landmarks, int points[]);
CascadeClassifier face_cascade;
CascadeClassifier eye_left_cascade;
CascadeClassifier eye_right_cascade;

int thresh = 200;
int max_thresh = 255;
const char* source_window = "Source image";
const char* corners_window = "Corners detected";

int main( int argc, const char** argv )
{

    String face_cascade_name = samples::findFile("../haarcascades/haarcascade_frontalface_alt.xml" );
    String eye_left_cascade_name = samples::findFile("../haarcascades/haarcascade_lefteye_2splits.xml");
    String eye_right_cascade_name = samples::findFile("../haarcascades/haarcascade_righteye_2splits.xml");

    if( !face_cascade.load( face_cascade_name ) )
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };
    
    if( !eye_left_cascade.load( eye_left_cascade_name ) )
    {
        cout << "--(!)Error loading eye left cascade\n";
        return -1;
    };
    
    if( !eye_right_cascade.load( eye_right_cascade_name ) )
    {
        cout << "--(!)Error loading eye right cascade\n";
        return -1;
    };

    VideoCapture capture("../sample_videos/bauka.mp4");
    if ( ! capture.isOpened() )
    {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }

    Mat frame;
    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }

        detectFaceEyesAndDisplay( frame );

        if( waitKey(10) == 27 )
        {
            break;
        }
    }
    return 0;
}

Point middlePoint(Point p1, Point p2) {
    float x = (float)((p1.x + p2.x) / 2);
    float y = (float)((p1.y + p2.y) / 2);
    Point p = Point(x, y);
    return p;
}

void detectFaceEyesAndDisplay( Mat frame )
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    std::vector<Rect> faces;
    face_cascade.detectMultiScale( frame_gray, faces );
    
    Mat faceROI = frame( faces[0] );
    Mat eye_left;
    Mat eye_right;
    Mat eye_left_gray;
    Mat eye_left_corner;
    Mat dst_norm, dst_norm_scaled;

    for ( size_t i = 0; i < faces.size(); i++ )
    {
        rectangle( frame,  Point(faces[i].x, faces[i].y), Size(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(255,0,0), 2 );

        Mat faceROI_gray = frame_gray( faces[i] );
        faceROI = frame( faces[i] );

        std::vector<Rect> eyes;
        std::vector<Vec3f> circles;
        eye_left_cascade.detectMultiScale(faceROI_gray, eyes);

        for ( size_t j = 0; j < eyes.size(); j++ )
        {
            rectangle(faceROI, Point(eyes[j].x, eyes[j].y + eyes[j].height/2), Size(eyes[j].x + eyes[j].width, eyes[j].y + eyes[j].height), Scalar(0, 255, 0), 2);
            eye_left = faceROI(eyes[0]);
            
            
           // cvtColor(eye_left, eye_left_gray, cv::COLOR_BGR2GRAY);
            //threshold(eye_left_gray, img_bw, 50.0, 255.0, THRESH_BINARY);
                        
            //cornerHarris(img_bw, eye_left_corner, 2, 3, 0.04);
            
            //normalize(eye_left_corner, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
            //convertScaleAbs( dst_norm, dst_norm_scaled );
//            for( int i = 0; i < dst_norm.rows ; i++ )
//            {
//                    for( int j = 0; j < dst_norm.cols; j++ )
//                    {
//                        if( (int) dst_norm.at<float>(i,j) > thresh )
//                        {
//                            circle( dst_norm_scaled, Point(j,i), 1,  Scalar(0), 1, 2, 0 );
//                        }
//                    }
//            }
                        
//            HoughCircles( eye_left_gray, circles, HOUGH_GRADIENT, 1, eye_left_gray.rows/8, 200, 100, 0, 0 );/// Draw the circles detected
//            for( size_t i = 0; i < circles.size(); i++ ) {
//                Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
//                int radius = cvRound(circles[i][2]);
//                // circle center
//                circle( eye_left, center, 3, Scalar(0,255,0), -1, 8, 0 );
//                // circle outline
//                circle( eye_left, center, radius, Scalar(0,0,255), 3, 8, 0 );
//            }
            cout <<  "Detected an eye" << std::endl ;
//            cornerHarris(eye_left_gray, eye_left_corner, 2, 3, 0.04);
//
//            normalize(eye_left_corner, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
//            convertScaleAbs( dst_norm, dst_norm_scaled );
//            for( int i = 0; i < dst_norm.rows ; i++ )
//            {
//                for( int j = 0; j < dst_norm.cols; j++ )
//                {
//                    if( (int) dst_norm.at<float>(i,j) > thresh )
//                    {
//                        circle( dst_norm_scaled, Point(j,i), 5,  Scalar(0), 2, 8, 0 );
//                    }
//                }
//            }

            
            cout <<  "Detected an eye" << std::endl ;
            
        }

    }
    
    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    
    imshow("Capture - Eye left", eye_left);


//    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create
//    imshow("Capture - Eye right", eye_right);
}
