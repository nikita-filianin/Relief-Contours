import torch
import cv2 as cv
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt

from UNet.model import UNet

class PngUtils():
    def __init__(self, conf: dict) -> None:
        self.conf = conf
        self.tsize = conf.TILE_SIZE
        self.thalf = self.tsize//2
        self.tquat = self.tsize//4


        self.device = conf.DEVICE
        self.mpath = conf.MODEL_PATH
        self.threshold = conf.THRESHOLD
        self.gsize = conf.GAP_SIZE
        self.isize = conf.ISLAND_SIZE
        self.imID = conf.imID
        self.image = plt.imread(conf.PNG_FPATH)

    def get_tiles(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Cuts .png image into overlaping 512x512 tiles
        Returns:
            np.array: array of tile corner (top left) coordinates (x, x+dx, y, y+dy)
            np.array: array of tiles
        """
        mask = (self.image > 0)

        y1_arr = np.arange(0, self.image.shape[0] - self.thalf, self.thalf).reshape(-1, 1)
        x1_arr = np.arange(0, self.image.shape[1] - self.thalf, self.thalf).reshape(-1, 1)

        grid = np.array(np.meshgrid(x1_arr, y1_arr)).T.reshape(-1, 2)
        grid = np.insert(grid, 1, grid[:,0] + self.tsize, axis = 1)
        grid = np.insert(grid, 3, grid[:,2] + self.tsize, axis = 1)

        coord_lst = []
        tiles_lst = []
        for i in range(grid.shape[0]):
            x0, x1, y0, y1 = grid[i][0], grid[i][1], grid[i][2], grid[i][3]
            msk = mask[y0:y1, x0:x1]
            if msk.sum() > 0:
                coord_lst.append(grid[i])
                tiles_lst.append(self.image[y0:y1, x0:x1])

        return(np.array(coord_lst), np.array(tiles_lst))
    
    def generate_coordinate_arrs(self, coord_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates coordinates of the submask with respect to the original image and the 512x512 tile it belongs to.
        Args:
            coord_arr (np.ndarray): Array of tile corner coordinates with respect to the original image.
        Returns:
            img_coord_arr (np.ndarray): Array of submask corner coordinates (x, x+dx, y, y+dy) with respect to the original image.
            tile_coord_arr (np.ndarray): Array of submask corner coordinates (x, x+dx, y, y+dy) with respect to the tile it belongs to.
        """

        img_coord_arr = coord_arr.copy()
        tile_coord_arr = np.tile([0, self.tsize, 0, self.tsize], (len(img_coord_arr), 1))
        for i in range(img_coord_arr.shape[1]):
            sign, val = (1, 0) if i%2 == 0 else (-1, img_coord_arr[:,i].max())
            
            img_coord_arr[:,i][img_coord_arr[:,i] != val] += sign*self.tquat
            tile_coord_arr[:,i][img_coord_arr[:,i] != val] += sign*self.tquat

        return (img_coord_arr, tile_coord_arr)


    def segment(self, coord_arr: np.ndarray, tile_arr: np.ndarray) -> np.ndarray:
        """
        Segments the image using the UNet model.
        Args:
            coord_arr (np.ndarray): Array of tile corner coordinates with respect to the original image.
            tile_arr (np.ndarray): Array of 512x512 tiles.
        Returns:
            mask (np.ndarray): Binary segmentaion mask for the whole image.
        """ 

        model = UNet(self.conf).to(self.device)
        model.load_state_dict(torch.load(self.mpath)['state_dict'])
        model.eval()
        print('Model loaded successfully')

        mask = np.zeros_like(self.image)

        img_coord_arr, tile_coord_arr = self.generate_coordinate_arrs(coord_arr)

        with torch.no_grad():
            for img_coord, tile_coord, tile in zip(img_coord_arr, tile_coord_arr, tile_arr):
                tile = torch.tensor(tile).float().to(self.device)
                tile = tile.unsqueeze(0).unsqueeze(0)

                predMask = model(tile).squeeze(0).squeeze(0)
                predMask = torch.relu(torch.sign(torch.sigmoid(predMask)-self.threshold))
                predMask = predMask.cpu().numpy().astype(np.uint8)

                ix0, ix1, iy0, iy1 = img_coord
                tx0, tx1, ty0, ty1 = tile_coord
                mask[iy0:iy1, ix0:ix1] = predMask[ty0:ty1, tx0:tx1]
        
        mask = mask > 0
        mask = morphology.remove_small_holes(mask, area_threshold=self.gsize)
        mask = morphology.remove_small_objects(mask, min_size=self.isize)
        return mask
    
    def edge_detect(self, mask: np.ndarray, cnt_density: np.uint8) -> np.ndarray:
        """
        Detect contours of the largest object in the binary mask
        Args:
            mask (np.ndarray): Binary mask.
            point_density (int): Gap between knot points along the contour (pixels); 
                                 the higher the value, the less points are kept, the smoother the contour.
        Returns:
            edge_points (np.ndarray): Array of edge points.
        """

        contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        if not contours:
            print('No contours found')
        else:
            edge_points = contours[0].reshape((-1, 2))
            idxs = np.arange(0, len(edge_points), cnt_density)
            edge_points = edge_points[idxs]
            
            return edge_points
        
