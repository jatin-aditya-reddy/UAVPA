# UAV for Precision Agriculture in a Modular Approach
This repository encloses my research work on UAV for precision agriculture that was published as a conference paper at IEEE UPCON 2020. The project uses an autonomous UAV, guided by the open source QGC app, and is capable of analysing plant health, and identifying insects and weeds in the field. The development is described in detail piecewise according to the phases in SDLC. 

## Introduction to the Project

* With the advancement in aerial robotics, the technology can be modified and developed to aid farmers in agriculture. 
* Common field problems involve analysing Plant Health status, and the presence of insects and weeds in the field. 
* Image processing tools, and concepts of machine learning are invoked and incorporated in the onto the UAV wherein a microcomputer (Raspberry Pi) is mounted.
* This project uses Raspberry Pi - NOIR Camera to get Near Infrared Spectrum for plant health analysis. Real time feed is sent to flask application server
* Calculations are done based on the NIR spectrum data to compute the Normalised Difference Vegetation Index for plants in the farm, and the health status is presented to the user on the flask application server. 

## Folder Structure
Folder             | Description
-------------------| -----------------------------------------
`1_Requirements`   | Documents detailing requirements and research
`2_Design`         | Documents specifying design details
`3_Implementation` | All code and documentation
`4_Test_plan`      | Documents with test plans and procedures
`5_Results`        | Showcases the results obtained in the project
`6_ImagesandVideos`| Demonstrates project output

## Contributors List and Summary

SF No. |  Name   |    Features    | Issuess Raised |Issues Resolved|No Test Cases|Test Case Pass
-------|---------|----------------|----------------|---------------|-------------|--------------
`22` | Jatin  | Hardware Test   | Nil     | Nil   |3   | Passed    
`01` | Abhay  | Autopilot   | Nil     | Nil   |4  | Passed
`22` | Jatin  | Health Analysis   | Nil     | Nil   |15   | Passed
`40` | Sakshi Verma  | I/W Detection  | Nil     | Nil   | 16  | Passed
`05` | Akhil Kumar  | Flask Server   | Nil     | Nil   |2   | Passed

## Challenges Faced and How Was It Overcome

1. Color mapping after calculating NDVI to output image on flask server - programming challenge
2. Dataset shortage - Solved by data mining

## Learning Resources

Here are some reference papers that were vital for the development of this project.

1. Rangel Daroya, Manuel Ramos, “NDVI image extraction of an agricultural land using an autonomous quad-copter with a filter-modified camera” International Conference on Control System, Computing and Engineering (ICCSCE), 2017
2. J. L. E. Honrado, D. B. Solpico, C. M. Favila, E. Tongson, G. L. Tangonan, N. J. C. Libatique, "UAV imaging with low-cost multispectral imaging system for precision agriculture applications", Global Humanitarian Technology Conference (GHTC) 2017 IEEE, pp. 1-7, 2017
3.	Pest Detection on UAV Imagery using a Deep Convolutional Neural Network *Yacine Bouroubi, Pierre Bugnet, Thuy Nguyen-Xuan, Claire Gosselin, Carl Bélec, Louis Longchamps and Philippe Vigneault In Proceedings of the 14th International Conference on Precision Agriculture (unpaginated, online). Monticello, IL: International Society of Precision Agriculture

## Credits

Project credits go to the team from IIIT Naya Raipur as available on the conference papers of IEEE UPCON 2020
A. K. Donka, J. A. R. Seerapu, S. Verma, A. Sao, G. Shukla and S. Tripathi, "Unmanned Aerial Vehicle for Precision Agriculture in a Modular Approach," 2020 IEEE 7th Uttar Pradesh Section International Conference on Electrical, Electronics and Computer Engineering (UPCON), 2020, pp. 1-5, doi: 10.1109/UPCON50219.2020.9376521.

