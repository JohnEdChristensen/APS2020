from manimlib.imports import *
import os
import pyclbr


class lat_point(Circle):
    CONFIG = {
    "radius" : 0.1,
    "stroke_width" : 3,
    "color" : WHITE,
    "fill_color" : WHITE,
    "fill_opacity" : 1.0,
    }
    def __init__(self, **kwargs):
        Circle.__init__(self, **kwargs)

class test(VectorScene):
    #A few simple shapes
    #Python 2.7 version runs in Python 3.7 without changes
    def construct(self):
        points = [2*x*RIGHT+2*y*UP
            for x in np.arange(-10,10,1)
            for y in np.arange(-10,10,1)
            ]     #List of vectors pointing to each grid point
        lat = []
        for p in points:
            lat.append(lat_point().shift(p))
        
        print(UP)
        print(RIGHT)
        draw_lat = VGroup(*lat)
        self.play(GrowFromCenter(draw_lat))
        #self.wait(5)
        v1 = Vector((0,2),color=GREEN)
        v2 = Vector((2,0),color=RED)
        self.add_vector(v1)
        self.add_vector(v2)
        

