from paraview.simple import *

import os

from paraview import servermanager



class para:

    @staticmethod
    def tryin():

        servermanager.Connect()

        connection = servermanager.Connect("amber", 10234, "destiny", "10235")

        # Sets servermanager.ActiveConnection to the connection object

        # servermanager.Disconnect() # disconnect


        # sphere = servermanager.

        # sphere =

        view = servermanager.CreateRenderView()

        repSphere = servermanager.CreateRepresentation(sphere, view)

        view.ResetCamera()

        view.StillRender()

if __name__ == '__main__':
    para.tryin()
