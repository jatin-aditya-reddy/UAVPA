# Requirements

## Introduction

* Hardware knowledge to develop a Drone (Quad-copter preferably), including soldering, interconnects on Powerdistribution Board, etc.
 
* Camera and Mini-Computer (Raspberry Pi) Mounts on the drone, and the payload lift capacity calculated to meet the weight constraints for safe flight

* Software Application to process the captured Near Infrared Spectrum on the NoIR Camera. 

* Development of the Flask application server to present the realtime feed to the system.

* Alternate Mount for an action camera to record video in SD/HD for insect and weed identification.

* Knowledge of QGroundControl (Open Source Application) integration to the flightcontroller (Pixhawk PX4) and manual operations (for emergencies and failsafe)

## Research

* Existing research (Cited below and in the conference paper, (link: https://ieeexplore.ieee.org/abstract/document/9376521), suggest the practicality of the product (if developed)

* It is possible to capture images and calculate the NIR band to estimate NDVI, this can be incorporated for a stream of images, i.e. video with defined FPS (Frames Per Second)

* Autonomous Aerial Vechiles are app-guided i.e. (Via PC or Smartphones), and the same technology can be incorporated with the proposed idea to help farmers operate such a complex system with ease.

* Drone Stability and product cost has been an industrial challenge owing to the physical complexities in the design.

Here are some reference papers that were vital for the development of this project.

1. Rangel Daroya, Manuel Ramos, “NDVI image extraction of an agricultural land using an autonomous quad-copter with a filter-modified camera” International Conference on Control System, Computing and Engineering (ICCSCE), 2017
2. J. L. E. Honrado, D. B. Solpico, C. M. Favila, E. Tongson, G. L. Tangonan, N. J. C. Libatique, "UAV imaging with low-cost multispectral imaging system for precision agriculture applications", Global Humanitarian Technology Conference (GHTC) 2017 IEEE, pp. 1-7, 2017
3.	Pest Detection on UAV Imagery using a Deep Convolutional Neural Network *Yacine Bouroubi, Pierre Bugnet, Thuy Nguyen-Xuan, Claire Gosselin, Carl Bélec, Louis Longchamps and Philippe Vigneault In Proceedings of the 14th International Conference on Precision Agriculture (unpaginated, online). Monticello, IL: International Society of Precision Agriculture

## Cost and Features
### *Cost*

* The software aspect of the system is free of cost, as it uses open source software and developed software from research accessible to everyone. 

* The hardware used in this research prototype involves basic drone components, Raspberry Pi 3B, Pixhawk PX4 FC, NOIR RPi camera and an action camera (SD/HD), the total cost is approxmiately INR 53000/- which is subject to change as per fluctuating market prices day by day. 


### *Features*

The features of the system are:
1. Autonomous Flight
2. Self Stablization (Hover)
3. Altitude Hold (Alt_Lock)
4. RTL (Return to launch i.e. failsafe)
5. Mobile App Control
6. Flight time ~ 20 minutes
7. Plant health analysis
8. Insect and weed detection


## Defining the System
![System Design](https://user-images.githubusercontent.com/84840612/120325783-a0811900-c305-11eb-93e3-8571a249c117.png)


## SWOT ANALYSIS
![SWOT](https://user-images.githubusercontent.com/84840612/120325808-a840bd80-c305-11eb-95fe-d4040fe5d6a9.jpg)


# 4W&#39;s and 1&#39;H

## Who:

The product will benefit farmers who would like to improve their crop yield in a harvest season. It can also provide an early insight into the cultivation season to help them make necessary changes early on.


## What:

A Unmanned Aerial Vehicle, for Precision Agriculture, capable of autonomous, app-controlled flight to identify insects, weeds and determine the health status of crops.

## When:

It can assist farmers to navigate across difficult portions of their farms, and more importantly complete the inspection and analysis in a short time span compared to manual inspection. It will save manpower additionally.

## Where:

The product can be put to use almost in every scenario involving farms, such as step farms, irrigation networks etc. The calculations and results obtained are accurate.

## How:

The UAV captures near Infrared radiations from the plants when they tap in sunlight during the day. It is a property for every plant to release these radiations upon illumination. A healthy plant is capable of reflecting almost entirely, while unhealthy plants can do so in traces. Other non-living entities and even animals and humans can't do so. The Normalized Difference Vegetation Index is identified from these values, and it is a ratio to enable us undestand plant health better. R-CNN deep learning model is used for weeds and insects and its idenfitication in the field. The results obtained can be seen in the flask application in real-time. 


# Detail requirements
## High Level Requirements:

|      ID          |Description                          |Status                         |
|----------------|-------------------------------|-----------------------------|
|HR_01|Hardware|Implemented|
|HR_02|Autopilot|Implemented|
|HR_03| Health Analysis|Implemented|
|HR_04|I/W Detection|Implemented|
|HR_05|Flask Server|Implemented|




##  Low level Requirements:
|      ID          |Description                          |  HLR_ID  |Status               |
|----------------|-------------------------------|----------|-----------------------------|
| LR_01   | Procurement     | HR_01   | Completed     |
| LR_02   | Frame Build     | HR_01   | Completed     |
| LR_03   | Soldering       | HR_01   | Completed     |
| LR_04   | Circuitry       | HR_01   | Completed     |
| LR_05   | Patch Update    | HR_02   | Completed     |
| LR_06   | Calibration     | HR_02   | Completed     |
| LR_07   | Telemetry Link  | HR_02   | Completed     |

|LR_01|Meter|HR_01|Implemented|
|LR_02|Centimeter|HR_01|Implemented|
|LR_03|Foot|HR_01|Implemented|
|LR_04|Inch|HR_01|Implemented|
|LR_05|Millimeter|HR_01|Implemented|
|LR_06|Square Meter|HR_02|Implemented|
|LR_07|Square Centimeter|HR_02|Implemented|
|LR_08|Square Foot|HR_02|Implemented|
|LR_09|MSquare Inch|HR_02|Implemented|
|LR_10|Square Yard|HR_02|Implemented|
|LR_11|Cubic Meter|HR_03|Implemented|
|LR_12|Cubic Centimeter|HR_03|Implemented|
|LR_13|Litre|HR_03|Implemented|
|LR_14|Millilitre|HR_03|Implemented|
|LR_15|Gallon (imperial)|HR_03|Implemented|
|LR_16|Kilogram|HR_04|Implemented|
|LR_17|Gram|HR_04|Implemented|
|LR_18|Ounce|HR_04|Implemented|
|LR_19|Tonne (metric)|HR_04|Implemented|
|LR_20|Pound|HR_04|Implemented|
|LR_21|Meter per Second|HR_05|Implemented|
|LR_22|Kilometer per Hour|HR_05|Implemented|
|LR_23|Miles per Hour|HR_05|Implemented|
|LR_24|Foot per Second|HR_05|Implemented|
|LR_25|Knot|HR_05|Implemented|
|LR_26|Hour|HR_06|Future|
|LR_27|Minute|HR_06|Future|
|LR_28|Second|HR_06|Future|
|LR_29|Celcius|HR_07|Future|
|LR_30|Farenheit|HR_07|Future|
|LR_31|Kelvin|HR_07|Future|
