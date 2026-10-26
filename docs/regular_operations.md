# Regular operations while running this project

This file contains my process to running this project, more as a human readable document to run the project

Disclaimer: work was done only on the robot with a red background. 

This document also assumes you read README.md and have installed the necessary libraries to run `aim_agent.py` on your laptop. 

## Run `robot_server.py` on Robot

- Ensure robot and computer are connected to the same WiFi 
- Upon starting the robot, open the terminal and wait a while. Nodes take a while to turn on 
- If you hear a beeping sound after ~50 seconds of turning on the robot, things should go well

Run `ros2 node list` on the robot's terminal to check the nodes running. If you got a long list of nodes, then it should be correct. More importantly, check for the following nodes: 
```
depth_cam/*
ros_robot_controller
```

Also, run `ros2 topic echo /ros_robot_controller/battery --once` to see if data appears. A number ranging from 9500 to 12000+ would appear. Beware that the robot would start beeping around 9500+, signalling **low battery**. 

After the nodes are turned on and the battery topic is giving an output, then you can run: 
```
export ROBOT_TOKEN=secret
python3 robot_server.py
```
And leave the terminal running. Go to the next section to run the other half of this project. 

* Note: Open another terminal to run `ros2 topic echo /ros_robot_controller/battery --once` periodically so that you do not get jumpscared when the robot runs out of battery

- Other things you would want to care about: 
    - Go to your computer browser, type `<robot-ip>:8080` and you will see a list of camera topics. `depth_cam/rgb/image_raw` is the topic where you get to see the livefeed (snapshot would show a picture) 
        - This helps a lot if you want a simple way to see what the robot camera sees 
    - If the live feed does not show anything, type `lsusb` to check that the camera is connected amongst the list of connections. 


## Run `aim_agent.py` on Computer

Unlike the robot that often gives a lot of issues, running `aim_agent.py` is a simpler task. 

Generally, running the following commands depending on which mode you want works fine:
```
# shoot mode 
python3 aim_agent.py --run-mode shoot --out runs

# track mode
python3 aim_agent.py --run-mode track --out runs_track --track-seconds 180
```
- shoot mode will end once target is aligned 
- track mode ends once the timer is up (your timer is `--track-seconds`, so vary the number to get desired duration)
- Add `--no-learn` if you don't want the agent to read past logs. Usually applied more for track mode because the object is "steadily moving", so the learning of pixels can be a bit iffy since offset between target and marker is due to the target moving also. 


_"The greatest limitation of this project is latency."_
