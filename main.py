from Configs.config_parser import Config

from Core.JP2Tools import GdalUtils
from Core.PngTools import PngUtils
from Core.utility import makeDirs, downloadJP2, save_results

import argparse
import time


def main(conf: dict):
    makeDirs(conf)
    downloadJP2(conf)

    jp2_img = GdalUtils(conf)
    if conf.STATS:
        jp2_img.get_stats()
    jp2_img.downscale()

    png_img = PngUtils(conf)
    coord_arr, tile_arr = png_img.get_tiles()
    mask = png_img.segment(coord_arr, tile_arr)
    edge_points = png_img.edge_detect(mask, conf.CNT_DENSITY)
    save_results(conf, edge_points)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Available parameters to enter')

    # General args
    parser.add_argument('--imID', metavar = ':', type=str, required = True, help='Image ID')
    parser.add_argument('--STORAGE_PATH', metavar = ':', type=str, help = 'Path to data storage')
    parser.add_argument('--CONFIG_PATH', metavar = ':', default = 'Configs/config.yaml', type=str, help = 'Path to .yaml config file')
    parser.add_argument('--MODEL_PATH', metavar = ':', type=str, help = 'Path to UNet .pth weights file')

    # Gdal args
    parser.add_argument('--BAND', metavar = ':', type=int, help='Band to read from the raster file')
    parser.add_argument('--RESIZE_FACTOR', metavar = ':', type=float,  help='Downscaling factor applied to the JP2 image')
    parser.add_argument('--TILE_SIZE', metavar = ':', type=int,  help='Size of the tiles cut from downscaled .png (pixels)')
    parser.add_argument('--STATS', action='store_true', help='Show gdal statistics?')

    # UNet args
    parser.add_argument('--NUM_CHANNELS', metavar = ':', type=int,  help='Number of channels in the input image')
    parser.add_argument('--NUM_CLASSES', metavar = ':', type=int,  help='Number of classes in the output mask')
    parser.add_argument('--THRESHOLD', metavar = ':', type=float, help='Threshold for the output mask; eveything below this value is set to 0, everything above to 1')

    # skimage.morphology.morphology & edge detection args
    parser.add_argument('--GAP_SIZE', metavar = ':', type=int,  help='Any gap smaller than this value will be filled (pixels)')
    parser.add_argument('--ISLAND_SIZE', metavar = ':', type=int,  help='Any island smaller than this value will be removed (pixels)')
    parser.add_argument('--CNT_DENSITY', metavar = ':', type=int,  help='Gap between knot points along the contour (pixels); the higher the value, the less points are kept, the smoother the contour')

    args = (parser.parse_args()).__dict__

    conf = Config(args['CONFIG_PATH']).getConfig(args)
  
    start_time = time.time()

    main(conf)
    
    end_time = time.time()
    processing_time = end_time - start_time
    print(f'Processing Time: {processing_time:.1f} seconds')