#!/usr/bin/env python
import rospy
import tf2_ros
import geometry_msgs.msg
from visualization_msgs.msg import Marker

x_lineup = -0.005

def publish_static_transform():
    """
    equivalent of 
    <node pkg="tf" type="static_transform_publisher" name="cylinder_broadcaster" args="0 0 0.155 0 0 -3.141517 /panda_hand /cylinder_origin 100"/>
    """
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    static_transform_stamped = geometry_msgs.msg.TransformStamped()

    static_transform_stamped.header.stamp = rospy.Time.now()
    static_transform_stamped.header.frame_id = "panda_hand"
    static_transform_stamped.child_frame_id = "cylinder_origin"

    static_transform_stamped.transform.translation.x = 0
    static_transform_stamped.transform.translation.y = 0
    static_transform_stamped.transform.translation.z = 0.155
    static_transform_stamped.transform.rotation.x = 0
    static_transform_stamped.transform.rotation.y = 0
    static_transform_stamped.transform.rotation.z = 0
    static_transform_stamped.transform.rotation.w = 1

    broadcaster.sendTransform(static_transform_stamped)
    rospy.sleep(1)  # Ensure the transform is broadcasted before the function ends

def publish_vertical_beams(pub):
    """
    2 vertical beams on the sides of the cylinder
    """
    # ===========================================
    # First vertical rectangular box
    box_marker1 = Marker()
    box_marker1.header.frame_id = "panda_link0"
    box_marker1.header.stamp = rospy.Time.now()
    box_marker1.ns = "obstacles"
    box_marker1.id = 1
    box_marker1.type = Marker.CUBE
    box_marker1.action = Marker.ADD
    box_marker1.pose.position.x = 0
    box_marker1.pose.position.y = 0.62  # Adjust position to the side of the cylinder
    box_marker1.pose.position.z = 0.2  # Adjust for vertical alignment
    box_marker1.pose.orientation.x = 0
    box_marker1.pose.orientation.y = 0
    box_marker1.pose.orientation.z = 0
    box_marker1.pose.orientation.w = 1

    box_marker1.scale.x = 0.05  # Width of the box
    box_marker1.scale.y = 0.05   # Depth of the box
    box_marker1.scale.z = 0.4   # Height of the box
    box_marker1.color.a = 1.0
    box_marker1.color.r = 0.8   # Light brown
    box_marker1.color.g = 0.52
    box_marker1.color.b = 0.25
    pub.publish(box_marker1)

    # ===========================================

    # Second vertical rectangular box
    box_marker2 = Marker()
    box_marker2.header.frame_id = "panda_link0"
    box_marker2.header.stamp = rospy.Time.now()
    box_marker2.ns = "obstacles"
    box_marker2.id = 2
    box_marker2.type = Marker.CUBE
    box_marker2.action = Marker.ADD
    box_marker2.pose.position.x = 0
    box_marker2.pose.position.y = 0.89  # Adjust position to the other side of the cylinder
    box_marker2.pose.position.z = 0.2  # Adjust for vertical alignment
    box_marker2.pose.orientation.x = 0
    box_marker2.pose.orientation.y = 0
    box_marker2.pose.orientation.z = 0
    box_marker2.pose.orientation.w = 1

    box_marker2.scale.x = 0.05  # Width of the box
    box_marker2.scale.y = 0.05  # Depth of the box
    box_marker2.scale.z = 0.4   # Height of the box
    box_marker2.color.a = 1.0
    box_marker2.color.r = 0.8   # Light brown
    box_marker2.color.g = 0.52
    box_marker2.color.b = 0.25
    pub.publish(box_marker2)

def publish_stick(pub):
    """
    single stick used for data collection
    """
    marker = Marker()
    marker.header.frame_id = "panda_link0"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "obstacles"
    marker.id = 0
    marker.type = Marker.CYLINDER
    marker.action = Marker.ADD
    marker.pose.position.x = 0.005
    marker.pose.position.y = 0.4
    marker.pose.position.z = 0.31
    marker.pose.orientation.x = 0.7071
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 0.7071
    
    marker.scale.x = 0.02  # Diameter in X (2 * radius)
    marker.scale.y = 0.02  # Diameter in Y (2 * radius)
    marker.scale.z = 0.4   # Height of the cylinder

    marker.color.a = 1.0
    marker.color.r = 0.55  # Red component
    marker.color.g = 0.36  # Green component
    marker.color.b = 0.36  # Blue component

    pub.publish(marker)

def publish_cross_easy(pub):
    """
    2 cylinders to form a cross
    """

    #first cylinder
    marker = Marker()
    marker.header.frame_id = "panda_link0"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "cross"
    marker.id = 0
    marker.type = Marker.CYLINDER
    marker.action = Marker.ADD
    marker.pose.position.x = x_lineup
    marker.pose.position.y = 0.55
    marker.pose.position.z = 0.31
    marker.pose.orientation.x = 0.7071
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 0.7071
    
    marker.scale.x = 0.015  # Diameter in X (2 * radius)
    marker.scale.y = 0.015  # Diameter in Y (2 * radius)
    marker.scale.z = 0.74   # Height of the cylinder

    marker.color.a = 1.0
    marker.color.r = 0.55  # Red component
    marker.color.g = 0.36  # Green component
    marker.color.b = 0.36  # Blue component

    pub.publish(marker)

    #second cylinder
    marker = Marker()
    marker.header.frame_id = "panda_link0"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "cross"
    marker.id = 1
    marker.type = Marker.CYLINDER
    marker.action = Marker.ADD
    marker.pose.position.x = x_lineup
    marker.pose.position.y = 0.42
    marker.pose.position.z = 0.31

    #rotate Rz by 90 degrees
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0.7071
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 0.707
    
    marker.scale.x = 0.015  # Diameter in X (2 * radius)
    marker.scale.y = 0.015  # Diameter in Y (2 * radius)
    marker.scale.z = 0.50   # Height of the cylinder

    marker.color.a = 1.0
    marker.color.r = 0.55  # Red component
    marker.color.g = 0.36  # Green component
    marker.color.b = 0.36  # Blue component

    pub.publish(marker)




def publish_obstacle():
    rospy.init_node('obstacle_publisher', anonymous=True)
    
    # Publish the static transform once
    publish_static_transform()

    pub = rospy.Publisher('/obstacle_marker', Marker, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():

        #publish the vertical beams
        publish_vertical_beams(pub)

        #publish obstacle
        publish_cross_easy(pub)
        
        

        rate.sleep()

if __name__ == '__main__':
    try:
        publish_obstacle()
    except rospy.ROSInterruptException:
        pass
