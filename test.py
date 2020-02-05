class Rectangle:
    # def __init__(self,  *args):
    #     self.length = args[0]
    #     self.width = args[1]
    def __init__ (self, **kwargs):
        self._Data=kwargs
    
    def GetTemplate_Location(self):
        return self._Data["template_loc"]
    def GetImage_Location(self):
        return self._Data["image_loc"]
    def GetTemplate_Threshold(self):
        return self._Data["template_thresh"]
    def GetNMS_Threshold(self):
        return self._Data["nms_thresh"]
    def GetOriginalImage(self):
        return self.image

    def area(self):
        return self.GetTemplate_Location() * self.GetImage_Location()

class Square(Rectangle):
    def __init__(self, **kwargs):
        self._Data=kwargs
    def GetNMS_Threshold(self):
        return self._Data["nms_thresh"]
    def run(self):    
        # super().__init__(template_loc=self.GetNMS_Threshold()[0], image_loc=self.GetNMS_Threshold()[1])
        super().__init__(template_loc=self.GetNMS_Threshold()[0])
        super().__init__(image_loc=self.GetNMS_Threshold()[1])
        print(super().area())

class VolumeMixin:
    def volume(self):
        return self.area() * self.height

class Cube(VolumeMixin, Square):
    def __init__(self, length):
        super().__init__(length)
        self.height = length

    def face_area(self):
        return super().area()

    def surface_area(self):
        return super().area() * 6

cube = Square(nms_thresh=[2,3,4])
cube.run()
# print(cube.area())