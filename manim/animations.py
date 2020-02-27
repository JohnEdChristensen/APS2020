from manimlib.imports import *
import os
import pyclbr
import numpy as np
from phenum.symmetry import get_spaceGroup



#lattice params
square = [2*UP,2*RIGHT]
hexagonal = [2*RIGHT,np.sqrt(3)*UP + RIGHT]
#oblique = [2*RIGHT+.2*UP,-np.sqrt(3)*UP + RIGHT]
oblique = [.75*UP + 1.5*RIGHT+LEFT +.75*UP,LEFT+.75*UP]
#print(b2)
b1 = hexagonal[0]
b2 = hexagonal[1]

v1 = Vector(b1,color=GREEN)
v2 = Vector(b2,color=RED)

bzv = np.array([
[ -5.92119e-16 , -1.1547 ,0],
[  1.0         , -0.57735,0],
[  1.0         ,  0.57735,0],
[  1.4803e-16  ,  1.1547 ,0],
[ -1.0         ,  0.57735,0],
[ -1.0         , -0.57735,0]])

class lat_create(VectorScene):
    #create a lattice and lattice vectors
    CONFIG ={
        "b1":hexagonal[0],
        "b2":hexagonal[1],
        "draw_lat":True,
        "show_vecs":True,
        "draw_vecs":True,
        "v1":Vector(b1,color=GREEN),
        "v2":Vector(b2,color=RED)
            }
    def start(self):

        points = [ self.b1*x+self.b2*y
            for x in np.arange(-10,10,1)
            for y in np.arange(-10,10,1)
            ]     #List of vectors pointing to each grid point
        lat = []

        for p in points:
            lat.append(Dot().shift(p))

        draw_lat = VGroup(*lat)

        if self.draw_lat:
            self.play(GrowFromCenter(draw_lat))
        else:
            self.add(draw_lat)
        #self.wait(5)
        if self.show_vecs:
            if self.draw_vecs:
                self.add_vector(self.v1)
                self.add_vector(self.v2)
            else:
                self.add_vector(self.v1,animate=False)
                self.add_vector(self.v2,animate=False)
    def construct(self):
        self.start()
class ob_create(lat_create):
    CONFIG ={
        "b1":oblique[0],
        "b2":oblique[1],
        "draw_lat":True,
        "show_vecs":True,
        "draw_vecs":True,
        "v1":Vector(oblique[0],color=GREEN),
        "v2":Vector(oblique[1],color=RED)
            }
class linear_comb(lat_create):
    #create a lattice and lattice vectors
    CONFIG={
        "b1":hexagonal[0],
        "b2":hexagonal[1],
        "draw_lat":False,
        "show_vecs":False,
        "draw_vecs":False,
        "v1":Vector(hexagonal[0],color=GREEN),
        "v2":Vector(hexagonal[1],color=RED)
            }
    def construct(self):
        self.start()
        #self.add(draw_lat)
        #self.add_vector(v1,animate=False)
        #self.add_vector(v2,animate=False)
        b1 = self.b1
        b2 = self.b2
        v1 = self.v1
        v2 = self.v2
        v1p = Vector(b1*2,color=v1.color)
        self.play(ApplyMethod(v2.shift,b1))
        self.wait()
        self.play(Transform(v1,v1p),ApplyMethod(v2.shift,b1))
        #self.play(ApplyMethod(v2.shift,b1))
        self.wait()
        v1p = Vector(-b1,color=v1.color)
        v2p = Vector(-b2,color=v2.color)
        #self.play(Transform(v1,v1p))
        self.play(Transform(v2,v2p.shift(-b1)),Transform(v1,v1p))
        self.wait()

        self.play(ApplyMethod(v2.shift,b1),Transform(v1,Vector(ORIGIN)))
        self.play(FadeOut(v2))

class lat_zoom(MovingCameraScene):
    def construct(self):

        points = [ b1*x+b2*y
                for x in np.arange(-35,35,1)
                for y in np.arange(-30,30,1)
                ]     #List of vectors pointing to each grid point

        lat = []
        for p in points:
            lat.append(Dot().shift(p))


        draw_lat = VGroup(*lat) 
        self.add(draw_lat)

        self.camera_frame.save_state()
        self.play(
                # Set the size with the width of a object
                self.camera_frame.set_width, 100,run_time=10
                )
        self.wait

        self.play(Restore(self.camera_frame))
def make_unit_cell(b1,b2):
        return  Polygon(ORIGIN,b1,b1+b2,b2,fill_opacity=.2)

class unit_cell(lat_create):
    CONFIG={
        "b1":hexagonal[0],
        "b2":hexagonal[1],
        "draw_lat":False,
        "show_vecs":True,
        "draw_vecs":True,
        "v1":Vector(hexagonal[0],color=GREEN),
        "v2":Vector(hexagonal[1],color=RED),
        "ucp":0
            }
    def uc(self):
        b1 = self.b1
        b2 = self.b2
        uc = make_unit_cell(b1,b2)
        self.ucp = uc
        self.play(ShowCreation(uc))
        self.wait()
        self.play(FadeOut(self.v1),FadeOut(self.v2))
    def construct(self):
        self.start()
        self.uc()

class tile(lat_create):
    CONFIG={
        "b1":hexagonal[0],
        "b2":hexagonal[1],
        "draw_lat":False,
        "show_vecs":False,
        "draw_vecs":False,
        "v1":0,
        "v2":0
            }
    def construct(self):
        self.start()
        b1 = self.b1
        b2 = self.b2

        uc = Polygon(ORIGIN,b1,b1+b2,b2,fill_opacity=.2)
        self.add(uc)
        tiles = []

        points = [ b1*x+b2*y
                for x in np.arange(-10,10,1)
                for y in np.arange(-10,10,1)
                ]     #List of vectors pointing to each grid point
        for p in points:
            if sum(abs(p)) != 0:
                tiles.append(Polygon(ORIGIN,b1,b1+b2,b2,fill_opacity=.2).shift(p))

        draw_tiles = VGroup(*tiles) 

        #self.play(ShowCreation(draw_tiles))
        self.play(*[ShowCreation(t) for t in draw_tiles])

def cut_line(p,op=ORIGIN,c=WHITE):
    line =DashedLine(op,p,positive_space_ratio = 0.8,color=c)
    line.rotate(np.pi/2).scale(20)

    return line

def make_bz(b1,b2):
    from julia.PEBSI import make_bz_2d
    bzv = make_bz_2d(np.transpose(np.vstack((b1[0:2],b2[0:2]))))
    bzv = np.insert(bzv,2,0.0,axis=1)
    print(bzv)
    bzp = Polygon(*bzv,color=YELLOW,fill_opacity =0.2 )
    return bzp
def make_bz_v(b1,b2):
    from julia.PEBSI import make_bz_2d
    bzv = make_bz_2d(np.transpose(np.vstack((b1[0:2],b2[0:2]))))
    bzv = np.insert(bzv,2,0.0,axis=1)
    return bzv
class bz(VectorScene):
    CONFIG={
            "b1":square[0],
            "b2":square[1],
            "speed":0.3
            }
    def construct(self):
        points = [ self.b1*x+self.b2*y
                for x in np.arange(-10,10,1)
                for y in np.arange(-10,10,1)
                ]     #List of vectors pointing to each grid point
        lat = []

        for p in points:
            lat.append(Dot().shift(p))

        draw_lat = VGroup(*lat)


        self.add(draw_lat)
        #line = Line(LEFT,UP).scale(10)
        points = [ self.b1*x+self.b2*y
                for x in np.arange(-1,2,1)
                for y in np.arange(-1,2,1)
                ]     #List of vectors pointing to each grid point
        points = [-self.b1,self.b2-self.b1,self.b2,self.b1,self.b1-self.b2,-self.b2]
        lines = []
        for p in points:
            if sum(p)!=0:
                lines.append(cut_line(p))
        cut_lines = []
        for p in points:
            if sum(p)!=0:
                line = Line(ORIGIN,p)
                dot = Dot(p,color=RED)
                self.play(ShowCreation(line),ShowCreation(dot),run_time=self.speed)
                #self.play(FocusOn(Dot(p,color=RED)))
                cut_lines.append(cut_line(p))
                self.play(ShowCreation(cut_lines[-1]),run_time=self.speed)
                self.play(FadeOut(line),FadeOut(dot),run_time=self.speed)

        bzp = make_bz(self.b1,self.b2)
        self.play(ShowCreation(bzp))
        self.play(*[FadeOut(l) for l in cut_lines])

        self.wait()

class hex_bz(bz):
    CONFIG={
            "b1":hexagonal[0],
            "b2":hexagonal[1],
            "speed":1
            }

class ob_bz(bz):
    CONFIG={
            "b1":oblique[0],
            "b2":oblique[1],
            "speed":0.3
            }

class bz_tile(VectorScene):
    CONFIG={
            "b1":square[0],
            "b2":square[1],
            }
    def construct(self):

        points = [ self.b1*x+self.b2*y
                for x in np.arange(-10,10,1)
                for y in np.arange(-10,10,1)
                ]     #List of vectors pointing to each grid point
        lat = []

        for p in points:
            lat.append(Dot().shift(p))

        draw_lat = VGroup(*lat)


        self.add(draw_lat)

        bzp = make_bz(self.b1,self.b2)
        self.add(bzp)

        points = [ self.b1*x+self.b2*y
                for x in np.arange(-10,10,1)
                for y in np.arange(-10,10,1)
                ]     #List of vectors pointing to each grid point
        tiles = []
        for p in points:
            if sum(abs(p)) !=0:
                tiles.append(make_bz(self.b1,self.b2).shift(p))

        draw_tiles = VGroup(*tiles) 

        #self.play(ShowCreation(draw_tiles))
        self.play(*[ShowCreation(t) for t in draw_tiles])

class hex_bz_tile(bz_tile):
    CONFIG={
            "b1":hexagonal[0],
            "b2":hexagonal[1],
            }

class ob_bz_tile(bz_tile):
    CONFIG={
            "b1":oblique[0],
            "b2":oblique[1],
            }
class bz_to_uc(Scene):
    def construct(self):
        yshift = 2.75
        xshift = 3
        hex_bzp = make_bz(hexagonal[0],hexagonal[1]).shift(yshift*UP+xshift*LEFT)
        sc_bzp = make_bz(square[0],square[1]).shift(xshift*LEFT)
        ob_bzp = make_bz(oblique[0],oblique[1]).shift(yshift*DOWN+xshift*LEFT)

        hex_uc = make_unit_cell(hexagonal[0],hexagonal[1]).shift(yshift*UP+xshift*RIGHT- hexagonal[0]/2 - hexagonal[1]/2)
        sc_uc = make_unit_cell(square[0],square[1]).shift(xshift*RIGHT- square[0]/2 - square[1]/2)
        ob_uc = make_unit_cell(oblique[0],oblique[1]).shift(yshift*DOWN+xshift*RIGHT- oblique[0]/2 - oblique[1]/2)

        hex_bzp.save_state()
        ob_bzp.save_state()
        sc_bzp.save_state()

        hex_uc.save_state()
        ob_uc.save_state()
        sc_uc.save_state()

        self.play(*[ShowCreation(t) for t in [hex_bzp,hex_uc]])
        self.play(*[ShowCreation(t) for t in [sc_bzp,sc_uc]])
        self.play(*[ShowCreation(t) for t in [ob_bzp,ob_uc]])
        self.wait(5)
        self.play(Transform(hex_bzp,hex_uc))
        self.play(Transform(sc_bzp,sc_uc))
        self.play(Transform(ob_bzp,ob_uc))
        self.wait()

        self.play(Restore(hex_bzp),Restore(ob_bzp),Restore(sc_bzp))


#(* Point group of a 2D Bravais lattinces, rotations are counterclockwise. *)
#rot[\[Theta]_]:={{Cos[\[Theta]],-Sin[\[Theta]]},{Sin[\[Theta]],Cos[\[Theta]]}};
#ref[\[Theta]_]:={{Cos[2 \[Theta]],Sin[2 \[Theta]]},{Sin[2 \[Theta]],-Cos[2\[Theta]]}};
#hexPointGroup[latAngle_]:=Join[rot@#&/@{0,\[Pi]/3,2\[Pi]/3,\[Pi],4\[Pi]/3,5\[Pi]/3},ref@#&/@{0,\[Pi]/6+latAngle,
#							\[Pi]/3+latAngle,\[Pi]/2+latAngle,2\[Pi]/3+latAngle,5\[Pi]/6+latAngle}];
#squarePointGroup[latAngle_]:=Join[rot@#&/@{0,\[Pi]/2,\[Pi],3\[Pi]/2},ref@#&/@{0,\[Pi]/4+latAngle,\[Pi]/2+latAngle,3\[Pi]/4+latAngle}];
#recPointGroup[latAngle_]:=Join[rot@#&/@{0,\[Pi]},ref@#&/@{0+\[Pi]+latAngle,\[Pi]/2+\[Pi]+latAngle}];
#centerRecPointGroup[latAngle_,angle_]:=Join[rot@#&/@{0,\[Pi]},ref@#&/@{angle/2+latAngle,angle/2+\[Pi]/2+latAngle}];
#obliquePointGroup=rot@#&/@{0,\[Pi]};

def rot(theta):
    return [[np.cos(theta),-np.sin(theta),0.0],[np.sin(theta),np.cos(theta),0.0],[0.0,0.0,0.0]]
def ref(theta):
    return [[np.cos(2*theta),np.sin(2*theta),0.0],[np.sin(2*theta),-np.cos(2*theta),0.0],[0.0,0.0,0.0]]
hexRotations = [0,np.pi/3,2*np.pi/3,np.pi,4*np.pi/3,5*np.pi/3]
hexRotationsTex = ["0","\\frac{\pi}{3}","2\\frac{\pi}{3}","\pi","4\\frac{\pi}{3}","5\\frac{\pi}{3}"]
hexReflectionsTex = ["0","\\frac{\pi}{6}","\\frac{\pi}{3}","\\frac{\pi}{2}","2\\frac{\pi}{3}","5\\frac{\pi}{6}"]
hexReflections = np.array([
        ref(0),
        ref(np.pi/6),
        ref(np.pi/3),
        ref(np.pi/2),
        ref(2*np.pi/3),
        ref(5*np.pi/6)])
hexAngles = np.array([
        0,
        np.pi/6,
        np.pi/3,
        np.pi/2,
        2*np.pi/3,
        5*np.pi/6])
hexPointGroup =np.array([
        rot(0.0),
        rot(np.pi/3),
        rot(2*np.pi/3),
        rot(np.pi),
        rot(4*np.pi/3),
        rot(5*np.pi/3),
        ref(0),
        ref(np.pi/6),
        ref(np.pi/3),
        ref(np.pi/2),
        ref(2*np.pi/3),
        ref(5*np.pi/6)])

squareRotationsTex = ["0","\\frac{\pi}{2}","3\\frac{\pi}{2}"]
squareReflectionsTex = ["0","\\frac{\pi}{4}","\\frac{\pi}{2}","3\\frac{\pi}{4}"]
squareRotations = [0,np.pi/2,3*np.pi/2]
squareReflections = np.array([
        ref(0),
        ref(np.pi/4),
        ref(np.pi/2),
        ref(3*np.pi/4)])

obRotationsTex = [0,"\pi"]
obReflectionsTex = []
obRotations = [0,np.pi]
obReflections = []
class ibz_setup(lat_create):
    CONFIG={
        "b1":hexagonal[0],
        "b2":hexagonal[1],
        "draw_lat":False,
        "show_vecs":False,
        "draw_vecs":False,
        "v1":0,
        "v2":0,
        "rots":hexRotations,
        "refs":hexReflections,
        "rotsTex":hexRotationsTex,
        "refsTex":hexReflectionsTex,
        "speed":1,
        "refGroup":0,
        "ibz":[[0.0,-1.1547,0.0],[0.5,-(1.1547-0.57735)/2 - 0.57735,0.0],ORIGIN],
        "bzv":bzv,
        "bzp":0,
        "rotGroup":0

            }

    def start2(self):
        self.start()
        b1 = self.b1
        b2 = self.b2

        self.bzv = make_bz_v(b1,b2)
        bzv = self.bzv
        
        self.bzp = Polygon(*bzv,color=YELLOW,fill_opacity =0.2 )
        bzp = self.bzp

        self.play(ShowCreation(bzp))

        #create op text

        rotT = [TextMobject("\\underline{Rotations}")]
        for e in self.rotsTex:
            rotT.append(TexMobject(e))
        rotGroup = VGroup(*rotT)
        rotGroup.arrange_submobjects(DOWN,aligned_edge=RIGHT)
        #rotGroup.shift(3*RIGHT)
        #rotGroup.add_background_rectangle()

        refT = [TextMobject("\\underline{Reflections}")]
        for e in self.refsTex:
            refT.append(TexMobject(e))
        refGroup = VGroup(*refT)
        refGroup.arrange_submobjects(DOWN,aligned_edge=RIGHT)

        pane = VGroup(rotGroup,refGroup)
        pane.arrange_submobjects(RIGHT,aligned_edge=UP)
        pane.add_background_rectangle()
        pane.shift(4*RIGHT)

        self.play(ShowCreation(pane))
        rotGroup = rotGroup[1:]
        self.rotGroup = rotGroup

#        self.play(ShowCreation(refGroup))
        refGroup = refGroup[1:]
        self.refGroup = refGroup

    def construct(self):
        self.start2()

class ibz(ibz_setup):
    CONFIG={
        "b1":hexagonal[0],
        "b2":hexagonal[1],
        "draw_lat":False,
        "show_vecs":False,
        "draw_vecs":False,
        "v1":0,
        "v2":0,
        "rots":hexRotations,
        "refs":hexReflections,
        "rotsTex":hexRotationsTex,
        "refsTex":hexReflectionsTex,
        "speed":1,
        "ibz":[[0.0,-1.1547,0.0],[0.5,-(1.1547-0.57735)/2 - 0.57735,0.0],ORIGIN],
            }
    def construct(self):
        self.start2()
        rotGroup = self.rotGroup
        refGroup = self.refGroup
        for e in rotGroup:
            print(e.get_tex_string())
            
        bzv = self.bzv
        v = bzv[0]

        cut_lines = []
        print(np.size(self.refs)+np.size(self.rots) )
        i = 0
        while(np.size(self.refs)+np.size(self.rots) != 1):
            stabs = []
            stabsTex = []
            v = bzv[i]
            vdot = Dot(v,color=RED)
            self.play(ShowCreation(vdot))
            j=0

            for angle in self.rots:
                if angle != 0:
                    vpdot = Dot(v,color=BLUE)
                    self.play(ShowCreation(vpdot),Indicate(rotGroup[j]))
                    self.play(Rotate(vpdot,angle,about_point=ORIGIN))
                    vp = vpdot.get_arc_center()
                    print(vp)
                    line = Line(v,vp)
                    self.play(ShowCreation(line),run_time=self.speed)
                    cut_lines.append(cut_line(vp,v,BLUE))
                    self.play(ShowCreation(cut_lines[-1]),run_time=self.speed)
                    self.play(FadeOut(line),FadeOut(vpdot),FadeOut(rotGroup[j]),run_time=self.speed/2)
                else: 
                    stabs.append(angle)
                    print("stab")
                    vpdot = Dot(v,color=RED) 
                    self.add(vpdot)
                    self.play(Flash(vpdot),Indicate(rotGroup[j]))
                    self.play(FadeOut(vpdot))
                j += 1
            self.rots = stabs

            stabs = []
            j = 0
            for r in self.refs:
                vp = r.dot(v)
                print(not np.allclose(v,vp))
                print(j)
                print(refGroup[j].get_tex_string())
                if not np.allclose(v,vp):
                    vpdot = Dot(vp,color=PURPLE)

                    ovdot = Dot(v,color=RED)

                    self.play(Transform(ovdot,vpdot),Indicate(refGroup[j]))
                    line = Line(v,vp)
                    self.play(ShowCreation(line),run_time=self.speed)
                    cut_lines.append(cut_line(vp,v,PURPLE))
                    self.play(ShowCreation(cut_lines[-1]),run_time=self.speed)
                    self.play(FadeOut(line),FadeOut(ovdot),FadeOut(refGroup[j]),run_time=self.speed/2)
                else:
                    stabs.append(r)
                    stabsTex.append(refGroup[j])
                    vpdot = Dot(v,color=PURPLE)
                    self.play(ShowCreation(vpdot))
                    self.play(Flash(vpdot),Indicate(refGroup[j]))
                    self.play(FadeOut(vpdot))
                    self.wait()
                    
                    print("stab")
                j += 1
            self.refs = stabs
            refGroup = stabsTex
            self.play(FadeOut(vdot))
            print(np.size(self.refs)+np.size(self.rots) )
            i +=1 
        self.play(ShowCreation(Polygon(*self.ibz,color=PURPLE,fill_opacity=.2 )))
        self.play(*[FadeOut(l) for l in cut_lines])

class square_ibz(ibz):
    CONFIG={
        "b1":square[0],
        "b2":square[1],
        "draw_lat":False,
        "show_vecs":False,
        "draw_vecs":False,
        "v1":0,
        "v2":0,
        "rots":squareRotations,
        "refs":squareReflections,
        "rotsTex":squareRotationsTex,
        "refsTex":squareReflectionsTex,
        "speed":1,
        "ibz":[[1.0,-1.0,0.0],[0.0,0.0,0.0],[1.0,0.0,0.0]]
            }
class ob_ibz(ibz):
    CONFIG={
        "b1":oblique[0],
        "b2":oblique[1],
        "draw_lat":False,
        "show_vecs":False,
        "draw_vecs":False,
        "v1":0,
        "v2":0,
        "rots":obRotations,
        "refs":obReflections,
        "rotsTex":obRotationsTex,
        "refsTex":obReflectionsTex,
        "speed":1,
        "ibz":[[-0.875, -0.125,0.0], [-0.625, -0.625,0.0],[0.125, -0.875,0.0],[-0.125, 0.875,0.0]]
        }
class textest(Scene):
    def construct(self):
        rotT = [TextMobject("\\underline{Rotations}")]
        for e in squareReflectionsTex:
            rotT.append(TexMobject(e))
        rotGroup = VGroup(*rotT)
        rotGroup.arrange_submobjects(DOWN,aligned_edge=LEFT)
        rotGroup.shift(3*RIGHT)
        self.play(ShowCreation(rotGroup))
        self.play(Indicate(rotGroup[1]))
        refT = [TextMobject("\\underline{Reflections}")]
        for e in squareRotationsTex:
            refT.append(TexMobject(e))
        refGroup = VGroup(*refT)
        refGroup.arrange_submobjects(DOWN,aligned_edge=LEFT)
        refGroup.shift(5*RIGHT)
        self.play(ShowCreation(refGroup))
        self.play(Indicate(refGroup[1]))
        self.play(FadeOut(refGroup[1]))

class sampling(unit_cell):
    CONFIG ={
        "b1":hexagonal[0],
        "b2":hexagonal[1],
        "draw_lat":True,
        "show_vecs":True,
        "draw_vecs":True,
        "v1":Vector(b1,color=GREEN),
        "v2":Vector(b2,color=RED)
        }
    def construct(self):
        self.start()
        self.uc()
        density = 6.0
        s1 = self.b1/density
        s2 = self.b2/density

        points = [ s1*x+s2*y
            for x in np.arange(0,density,1)
            for y in np.arange(0,density,1)
            ]     #List of sampling points

        sample = []
        for p in points:
            sample.append(Dot(radius=0.05,color=RED).shift(p))

        draw_sample = VGroup(*sample)
        self.play(ShowCreation(draw_sample))
        #self.play(FadeOut(draw_sample))
        print(self.ucp)
        self.play(FadeOut(self.ucp))

        self.play(ShowCreation(Polygon(*bzv,color=YELLOW,fill_opacity = 0.2)))

        bpoints = [ [s1*x+s2*y,s1*x+s2*y-b1,s1*x+s2*y-b1-b2,s1*x+s2*y-b2]
            for x in np.arange(0,density,1)
            for y in np.arange(0,density,1)
            ]     #List of sampling points
        bsample = []
        for ps in bpoints:
            distance = 100
            closest_point = 0
            for p in ps:
                d = np.linalg.norm(p)
                if d < distance -.000001:
                    distance = d
                    closest_point = p
            bsample.append(Dot(radius=0.05,color=GREEN).shift(closest_point))

        draw_bsample = VGroup(*bsample)
        self.play(Transform(draw_sample,draw_bsample))

class nu_sampling(lat_create):
    CONFIG ={
        "b1":hexagonal[0],
        "b2":hexagonal[1],
        "draw_lat":False,
        "show_vecs":False,
        "draw_vecs":False,
        "v1":Vector(b1,color=GREEN),
        "v2":Vector(b2,color=RED)
        }
    def construct(self):
        self.start()
        density = 6.0
        s1 = self.b1/density
        s2 = self.b2/density

        points = [ s1*x+s2*y
            for x in np.arange(0,density,1)
            for y in np.arange(0,density,1)
            ]     #List of sampling points

        sample = []
        for p in points:
            sample.append(Dot(radius=0.05,color=RED).shift(p))

        draw_sample = VGroup(*sample)
        #self.add(draw_sample)

        self.add(Polygon(*bzv,color=YELLOW,fill_opacity = 0.2))

        bpoints = [ [s1*x+s2*y,s1*x+s2*y-b1,s1*x+s2*y-b1-b2,s1*x+s2*y-b2]
            for x in np.arange(0,density,1)
            for y in np.arange(0,density,1)
            ]     #List of sampling points
        bsample = []
        for ps in bpoints:
            distance = 100
            closest_point = 0
            for p in ps:
                d = np.linalg.norm(p)
                if d < distance -.000001:
                    distance = d
                    closest_point = p
            bsample.append(Dot(radius=0.05,color=GREEN).shift(closest_point))

        density = 12.0
        s1 = self.b1/density
        print(s1)
        s2 = self.b2/density
        #print(s2)

        bpoints2 = [s1*x+s2*y
            for x in np.arange(0,density,1)
            for y in np.arange(0,density,1)
            ]     #List of sampling points
        bsample2 = []
        #print(bpoints2)
        for p in bpoints2:
            #print(p)
            d = np.linalg.norm(p)
            if d < 1:
                bsample2.append(Dot(radius=0.05,color=GREEN).shift(p))

        draw_bsample = VGroup(*bsample)
        draw_bsample2 = VGroup(*bsample2)
        self.add(draw_bsample)
        self.play(ShowCreation(draw_bsample2))

class rectangle_rule(GraphScene):
    CONFIG = {
            "x_min":-4,
            "x_max":4,
        "y_max": 4,
        "y_min": -4,
        "y_axis_height": 4,
        "graph_origin":ORIGIN,
        "init_dx":0.5,
    }
    def construct(self):
        self.setup_axes()
        eq = "\int_{-3}^3 \\frac{1}{5} x^4 - \\frac{3}{2} x^2 dx "
        funcTex = TexMobject(eq).shift(-3*UP, -3.1*RIGHT)
        def func(x):
            return -1.5*x**2 + .2*x**4

        kwargs = { "x_min" : -3,
                "x_max" : 3,
                "fill_opacity" : 0.75,
                "stroke_width" : 0.25,
                }
        graph=self.get_graph(func,x_min=-4,x_max=4)
        #self.add(graph,riemann_rectangles)
        flat_rectangles = self.get_riemann_rectangles(
                self.get_graph(lambda x : 0),
                dx=self.init_dx,
                start_color=invert_color(PURPLE),
                end_color=invert_color(ORANGE),
                **kwargs
                )
        riemann_rectangles_list = self.get_riemann_rectangles_list(
                graph,
                5,
                max_dx=self.init_dx,
                power_base=2,
                start_color=PURPLE,
                end_color=ORANGE,
                **kwargs
                )
        self.play(ShowCreation(graph),ShowCreation(funcTex))
        # Show Riemann rectangles
        self.play(ReplacementTransform(flat_rectangles,riemann_rectangles_list[0]))
        self.wait()
        appTex = TexMobject("\\approx").shift(DOWN*3)
        nTex = TexMobject("N = 12" ).shift(UP*3)
        self.play(ShowCreation(nTex))
        errorTex = TexMobject(" \\text{Error} = ").shift(DOWN*3+RIGHT*3.5)
        self.play(ShowCreation(appTex))
        i = 0
        actual  = -7.56
        dx = float(self.init_dx) / 2**i
        app = rectint(func,-3,3,dx)
        error = (app -  actual)/ actual * 100
        self.play(Transform(appTex,TexMobject( "\\approx %8.4f" % (app)).shift(DOWN*3)),
        Transform(nTex,TexMobject("N = %i"%(6.0/dx)).shift(UP*3)),
        Transform(errorTex,TexMobject( "\\text{Error} =  %8.2f\\%%" % (error)).shift(DOWN*3+RIGHT*3.5)))
        self.wait()
        i = 1
        for r in range(1,len(riemann_rectangles_list)):
            dx = float(self.init_dx) / 2**i
            app = rectint(func,-3,3,dx)
            error = (app -  actual)/ actual * 100

            self.transform_between_riemann_rects(
                    riemann_rectangles_list[r-1],
                    riemann_rectangles_list[r],
                    replace_mobject_with_target_in_scene = True,
                    )
            i+=1
            self.play(Transform(appTex,TexMobject( "\\approx %8.4f" % (app)).shift(DOWN*3)),
            Transform(nTex,TexMobject("N = %i"%(6.0/dx)).shift(UP*3)),
            Transform(errorTex,TexMobject( "\\text{Error} =  %8.2f\\%%" % (error)).shift(DOWN*3+RIGHT*3.5)))
            self.wait()

class rectangle_rule2(GraphScene):
    CONFIG = {
            "x_min":-4,
            "x_max":4,
        "y_max": 4,
        "y_min": -4,
        "y_axis_height": 4,
        "graph_origin":ORIGIN,
        "init_dx":0.5,
    }
    def construct(self):
        self.setup_axes()
        eq = "2*\int_{0}^3 \\frac{1}{5} x^4 - \\frac{3}{2} x^2 dx "
        funcTex = TexMobject(eq).shift(-3*UP, -3.4*RIGHT)
        def func(x):
            return -1.5*x**2 + .2*x**4

        kwargs = { "x_min" : 0,
                "x_max" : 3,
                "fill_opacity" : 0.75,
                "stroke_width" : 0.25,
                }
        graph=self.get_graph(func,x_min=-4,x_max=4)
        #self.add(graph,riemann_rectangles)
        flat_rectangles = self.get_riemann_rectangles(
                self.get_graph(lambda x : 0),
                dx=self.init_dx,
                start_color=invert_color(PURPLE),
                end_color=invert_color(ORANGE),
                **kwargs
                )
        riemann_rectangles_list = self.get_riemann_rectangles_list(
                graph,
                5,
                max_dx=self.init_dx,
                power_base=2,
                start_color=PURPLE,
                end_color=ORANGE,
                **kwargs
                )
        self.play(ShowCreation(graph),ShowCreation(funcTex))
        # Show Riemann rectangles
        self.play(ReplacementTransform(flat_rectangles,riemann_rectangles_list[0]))
        appTex = TexMobject("\\approx").shift(DOWN*3)
        nTex = TexMobject("N = 6" ).shift(UP*3)
        self.play(ShowCreation(nTex))
        errorTex = TexMobject(" \\text{Error} = ").shift(DOWN*3+RIGHT*3.5)
        self.play(ShowCreation(appTex))
        i = 0
        actual  = -7.56
        dx = float(self.init_dx) / 2**i
        app = 2*rectint(func,0,3,dx)
        error = (app -  actual)/ actual * 100
        self.play(Transform(appTex,TexMobject( "\\approx %8.4f" % (app)).shift(DOWN*3)),
        Transform(nTex,TexMobject("N = %i"%(3.0/dx)).shift(UP*3)),
        Transform(errorTex,TexMobject( "\\text{Error} =  %8.2f\\%%" % (error)).shift(DOWN*3+RIGHT*3.5)))
        i = 1
        for r in range(1,len(riemann_rectangles_list)):
            dx = float(self.init_dx) / 2**i
            app = 2*rectint(func,0,3,dx)
            error = (app -  actual)/ actual * 100

            self.transform_between_riemann_rects(
                    riemann_rectangles_list[r-1],
                    riemann_rectangles_list[r],
                    replace_mobject_with_target_in_scene = True,
                    )
            i+=1
            self.play(Transform(appTex,TexMobject( "\\approx %8.4f" % (app)).shift(DOWN*3)),
            Transform(nTex,TexMobject("N = %i"%(3.0/dx)).shift(UP*3)),
            Transform(errorTex,TexMobject( "\\text{Error} =  %8.2f\\%%" % (error)).shift(DOWN*3+RIGHT*3.5)))
def rectint(f,a,b,i):
    cumulative_area=0

    a=float(a)
    b=float(b)

    trailing_x=a
    leading_x=a+i

    while (a<=leading_x<=b) or (a>=leading_x>=b):
        area=f((trailing_x+leading_x)/2)*i
        cumulative_area+=area

        leading_x+=i
        trailing_x+=i

    return cumulative_area

class sym(ibz_setup):
    CONFIG={
        "b1":hexagonal[0],
        "b2":hexagonal[1],
        "draw_lat":False,
        "show_vecs":False,
        "draw_vecs":False,
        "v1":0,
        "v2":0,
        "rots":hexRotations,
        "refs":hexReflections,
        "rotsTex":hexRotationsTex,
        "refsTex":hexReflectionsTex,
        "speed":1,
        "ibz":[[0.0,-1.1547,0.0],[0.5,-(1.1547-0.57735)/2 - 0.57735,0.0],ORIGIN],
            }
    def construct(self):
        self.start2()
        rotGroup = self.rotGroup
        refGroup = self.refGroup
            
        bzv = self.bzv
        bzp =  self.bzp
        v = bzv[0]

        i = 0
        for angle in self.rots:
            self.play(Rotate(bzp,angle,about_point=ORIGIN),Indicate(rotGroup[i]),run_time=2)
            self.wait()
            i+=1
        i=0
        for r in hexAngles:
            ax = np.array([np.cos(r),np.sin(r),0.0])
            self.play(Rotate(bzp,np.pi,axis = ax,about_point=ORIGIN),Indicate(refGroup[i]),run_time=1)
            self.wait()
            i+=1


