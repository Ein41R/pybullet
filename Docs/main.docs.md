
The idea is this:

we have pybulled create a simulation environment,
We controll the robot model (ur10e.urdf) within the simulation via ROS2
Inside that sits bdh which we will see how it will perform.

opt: experimenting with lidar on Iphone 12 Pro onward using sensorstream

We use Universal_Robots official description rpository (though forked)
Also the bulled3 assets library
with bdh from pathway.

Essentially BDH is a transformer that has its attention mechanism replaced with 
a tensor acting as a assiciative memory between "neurons". I like that apporach
personally, therefore Im experimenting with that.

Additionally, I believe pathway optimized bdh very well with the encoder decoder architecture,
which should be optimized enough to run on my old budget PC from highschool. 
(PCs getting to expencive nowadays)