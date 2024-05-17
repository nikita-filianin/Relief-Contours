import os
import numpy as np
from osgeo import gdal
gdal.UseExceptions()

class GdalUtils():
    def __init__(self, conf: dict) -> None:
        self.conf = conf
        self.rf = conf.RESIZE_FACTOR
        self.tsize = conf.TILE_SIZE
        self.bnum = conf.BAND
        self.pngpath = conf.PNG_FPATH

        self.dataset = gdal.Open(conf.JP2_FPATH, gdal.GA_ReadOnly)
        self.geotransform = self.dataset.GetGeoTransform()
        self.band = self.dataset.GetRasterBand(1)

        self.min_val, self.max_val = self.band.ComputeRasterMinMax(True)
        self.min_val, self.max_val = int(self.min_val), int(self.max_val)

        self.XSize, self.YSize, self.band_num = self.dataset.RasterXSize, self.dataset.RasterYSize, self.dataset.RasterCount

    def get_stats(self):
        """
        Prints basic information about the image
        """
        print("Driver: {}/{}".format(self.dataset.GetDriver().ShortName,
                                    self.dataset.GetDriver().LongName))
        print("Size is {} x {} x {}".format(self.XSize,
                                            self.YSize,
                                            self.band_num))
        print("Projection is {}".format(self.dataset.GetProjection()))

        print("Origin = ({}, {})".format(self.geotransform[0], self.geotransform[3]))
        print("Pixel Size = ({}, {})".format(self.geotransform[1], self.geotransform[5]))

        print("Band Type={}".format(gdal.GetDataTypeName(self.band.DataType)))
        print("Min={}, Max={}".format(self.min_val, self.max_val))

    def downscale(self, save_format: str = 'png', rm_xml: bool = True) -> None:
        """
        Downscales the .JP2 image and saves it as .png
        Args:
            save_format (str): format of the saved image
            rm_xml (bool): whether to remove .aux.xml file
        Returns:
            None: saves the image as .png
        """
        if not os.path.exists(self.pngpath):
            W = np.round((self.XSize*self.rf)/self.tsize).astype(int)*self.tsize
            H = np.round((self.YSize*self.rf)/self.tsize).astype(int)*self.tsize
            gdal.Translate(self.pngpath, self.dataset, bandList = [self.bnum], width = W, height = H, \
                        format = save_format, scaleParams = [[self.min_val, self.max_val, self.min_val, 255]], outputType=gdal.GDT_Byte)
            
            if rm_xml:
                os.remove(f'{self.pngpath}.aux.xml')
        else:
            print(f'{self.pngpath} already exists')
