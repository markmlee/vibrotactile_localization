#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker

def publish_obstacle():
    rospy.init_node('obstacle_publisher', anonymous=True)
    pub = rospy.Publisher('/obstacle_marker', Marker, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        marker = Marker()
        marker.header.frame_id = "panda_link0"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "obstacles"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = 0
        marker.pose.position.y = 0.496
        marker.pose.position.z = 0.31
        marker.pose.orientation.x = 0.7071
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 0.7071
        
        marker.scale.x = 0.02  # Diameter in X (2 * radius)
        marker.scale.y = 0.02  # Diameter in Y (2 * radius)
        marker.scale.z = 0.45   # Height of the cylinder

        marker.color.a = 1.0
        marker.color.r = 0.55  # Red component
        marker.color.g = 0.36  # Green component
        marker.color.b = 0.36  # Blue component

        pub.publish(marker)

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
        box_marker1.pose.position.y = 0.3  # Adjust position to the side of the cylinder
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
        box_marker2.pose.position.y = 0.8  # Adjust position to the other side of the cylinder
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


        rate.sleep()

if __name__ == '__main__':
    try:
        publish_obstacle()
    except rospy.ROSInterruptException:
        pass
