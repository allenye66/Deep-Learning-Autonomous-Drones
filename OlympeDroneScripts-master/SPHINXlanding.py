# -*- coding: UTF-8 -*-

import olympe
from olympe.messages.ardrone3.Piloting import Landing
drone = olympe.Drone("10.202.0.1")
drone.connection()
drone(Landing()).wait()
drone.disconnection()

