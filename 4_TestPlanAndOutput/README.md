# Test Plan and Test Output

## Flight test (High level test)
* Checking for all calibration tests for success on QGC app - Successful
* Flight test: lift off, attributes (throttle, yaw, pitch, roll) , RTL - Successful

## Software test - Plant Health Analysis (High Level Test)
* NDVI calculation for plants in feed - passed
* Live Stream on Flask app on Local Host - passed

## Insect and Weed Detection (High Level Test)
* Model accuracy - 89.5% - passed
* Test cases - passed

## Unit Test (Low level test)
* Feed Stability: action camera to identify stability in the mount - Requirement based test - successful
* Constant framerate of stream (30fps) - requirement based test - passed
* Hardware aspects - passed
* 
* 
* volume options function - Requirement based test - Passing
* mass options function - Requirement based test - Passing
* speed options function - Requirement based test - Passing
* file extraction function - Scenario based test - Passing
* conversion function - Scenario based test - Passing
* print output table - - Scenario based test - Passing
