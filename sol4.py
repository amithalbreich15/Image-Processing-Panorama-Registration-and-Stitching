# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API
# is unchanged

import random

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass, map_coordinates
import shutil
from imageio import imwrite
import sol4_utils

PATCH_SIZE = 7
SECOND_MAX = -2
SQUARE = 2
K = 0.04
PYR_TRANS_FACTOR = 0.25
LEVEL_ZERO = 0
LEVEL_TWO = 2
KERNEL_SIZE = 3
DESC_DEFAULT_RADIUS = 3
RADIUS = 20


def harris_corner_detector(im):
    """
  Detects harris corners.
  Make sure the returned coordinates are x major!!!
  :param im: A 2D array representing an image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates
   of the ith corner points.
  """
    der_kernel = np.array([[1, 0, -1]])
    i_x = convolve(im, der_kernel)
    i_y = convolve(im, der_kernel.T)
    i_x_x_blurred = sol4_utils.blur_spatial(i_x ** 2, KERNEL_SIZE)
    i_x_y_blurred = sol4_utils.blur_spatial(i_x * i_y, KERNEL_SIZE)
    i_y_y_blurred = sol4_utils.blur_spatial(i_y ** 2, KERNEL_SIZE)
    R = (i_x_x_blurred * i_y_y_blurred - i_x_y_blurred ** 2) - \
        K * (i_x_x_blurred + i_y_y_blurred) ** 2
    max_bool = non_maximum_suppression(R)
    return np.argwhere(max_bool.T).astype(int)


def sample_one_descriptor(point_coords, im, desc_rad):
    """
    This function operates samples descriptor for one point
    :param pos: the point to sample
    :param im: the image from which the points was taken
    :param desc_rad: "Radius" of descriptors to compute.
    :return: a 2D array with shape (K,K) containing the descriptor for the
     given point.
    """
    coord_x = point_coords[0] - desc_rad
    coord_y = point_coords[1] - desc_rad
    patch_y, patch_x = np.mgrid[coord_y:coord_y + 2 * desc_rad + 1,
                       coord_x:coord_x + 2 * desc_rad + 1]
    window = map_coordinates(im, [patch_y, patch_x], order=1, prefilter=False)
    mean = np.mean(window)
    window = window - mean
    norm = np.linalg.norm(window)
    if norm == 0:
        return np.zeros_like(window)
    return window / norm


def sample_descriptor(im, pos, desc_rad):
    """
  Samples descriptors at the given corners.
  :param im: A 2D array representing an image.
  :param pos: An array with shape (N,2), where pos[i,:] are the [x,y]
  coordinates of the ith corner point.
  :param desc_rad: "Radius" of descriptors to compute.
  :return: A 3D array with shape (N,K,K) containing the ith descriptor at
   desc[i,:,:].
  """
    pyr0_mapped_pos = PYR_TRANS_FACTOR * pos
    return np.apply_along_axis(sample_one_descriptor, 1, pyr0_mapped_pos,
                               im, desc_rad)


def find_features(pyr):
    """
  Detects and extracts feature points from a pyramid.
  :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
  :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row
                 found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
  """
    points = spread_out_corners(pyr[LEVEL_ZERO], PATCH_SIZE, PATCH_SIZE,
                                RADIUS)
    desc = sample_descriptor(pyr[LEVEL_TWO], points, DESC_DEFAULT_RADIUS)
    return [points, desc]


def match_features(desc1, desc2, min_score):
    """
  Return indices of matching descriptors.
  :param desc1: A feature descriptor array with shape (N1,K,K).
  :param desc2: A feature descriptor array with shape (N2,K,K).
  :param min_score: Minimal match score.
  :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices
               in desc1.
              2) An array with shape (M,) and dtype int of matching indices
              in desc2.
  """
    desc1_flat = desc1.reshape(desc1.shape[0], -1)
    desc2_flat = desc2.reshape(desc2.shape[0], -1)
    S = np.dot(desc1_flat, desc2_flat.T)
    second_max_scores_rows = np.partition(S, SECOND_MAX, axis=1)[:, SECOND_MAX]
    second_max_scores_cols = np.partition(S, SECOND_MAX, axis=0)[SECOND_MAX, :]
    rows = S >= second_max_scores_rows.reshape((len(S), 1))
    cols = S >= second_max_scores_cols.reshape(1, len(S[0]))
    valid_scores = S > min_score
    matches = np.logical_and(np.logical_and(rows, cols), valid_scores)
    return np.where(matches)


def apply_homography(pos1, H12):
    """
  Apply homography to inhomogenous points.
  :param pos1: An array with shape (N,2) of [x,y] point coordinates.
  :param H12: A 3x3 homography matrix.
  :return: An array with the same shape as pos1 with [x,y] point
  coordinates obtained from transforming pos1 using H12.
  """
    ones_vector = np.ones((pos1.shape[0], 1))
    ones_added_coords = np.hstack((pos1, ones_vector))
    after_homography = np.matmul(ones_added_coords, H12.T)
    return after_homography[:, :2] / after_homography[:, 2:3]


def compute_inliers(points1, points2, H, inlier_tol):
    """
     Computes homography between two sets of points using RANSAC.
     :param points1: An array with shape (N,2) containing N rows of [x,y]
     coordinates
      of matched points in image 1.
     :param points1: An array with shape (N,2) containing N rows of [x,y]
      coordinates
      of matched points in image 2.
     :param H Current Homography
     :param inlier_tol: inlier tolerance threshold.
     """
    # Transform points1 using the homography
    points1_transformed = apply_homography(points1, H)

    # Compute distance between the transformed points and the points in points2
    distances = np.power(np.linalg.norm(points1_transformed - points2, axis=1),
                         SQUARE)

    # Return the indices of the inliers
    return np.argwhere(np.array(distances < inlier_tol))


def ransac_homography(points1, points2, num_iter, inlier_tol,
                      translation_only=False):
    """
  Computes homography betX  ween two sets of points using RANSAC.
  :param points1: An array with shape (N,2) containing N rows of [x,y]
  coordinates
   of matched points in image 1.
  :param points1: An array with shape (N,2) containing N rows of [x,y]
  coordinates
   of matched points in image 2.
  :param num_iter: Number of RANSAC iterations to perform.
  :param inlier_tol: inlier tolerance threshold.
  :param translation_only: see estimate rigid transform
  :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of
                  inlier matches found.
  """
    # Number of points
    N = points1.shape[0]

    # Initialize variables to store the best homography and inliers
    best_H = None
    best_inliers = None
    for i in range(num_iter):
        # Randomly select 2 point pairs or 1 if translation only
        if not translation_only:
            indices = np.random.choice(N, size=2, replace=False)
        else:
            indices = np.random.choice(N, size=1, replace=False)
        pts1 = points1[indices]
        pts2 = points2[indices]

        # Compute homography between the selected points
        H = estimate_rigid_transform(np.array(pts1), np.array(pts2),
                                     translation_only)

        # Compute inliers by checking the distance between the transformed
        # points and the original points
        inliers = compute_inliers(points1, points2, H, inlier_tol)
        inliers = inliers.reshape(np.size(inliers), )

        # If this homography has more inliers, it is the new best homography
        if best_inliers is None or np.size(inliers) > np.size(best_inliers):
            best_H = H
            best_inliers = inliers
    return best_H, best_inliers


def display_matches(im1, im2, points1, points2, inliers):
    """
  Dispalay matching points.
  :param im1: A grayscale image.
  :param im2: A grayscale image.
  :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates
   of matched points in im1.
  :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates
   of matched points in im2.
  :param inliers: An array with shape (S,) of inlier matches.
  """
    extended_image = np.hstack((im1, im2))

    # fit image2 coordinates
    aligned_points2 = points2
    aligned_points2[:, 0] += im1.shape[1]

    # plot the image with the inliers and outliers
    plot_image_with_liers(aligned_points2, extended_image, inliers, points1)


def plot_image_with_liers(aligned_points2, extended_image, inliers, points1):
    """
    Helper function for display_matches - creates the figure and plots
    relevant data such as inliers and outliers and edges.
    :param aligned_points2: points from second image aligned to correct pos.
    :param extended_image:image1 and image2 presented next to each other.
    :param inliers: inliers to show
    :param points1: points from first image
    :return:
    """
    outliers1 = np.delete(points1, inliers, axis=0)
    outliers2 = np.delete(aligned_points2, inliers, axis=0)
    plt.figure()
    plt.imshow(extended_image, cmap='gray')
    # plot the corners with red markings
    plt.scatter((outliers1[:, 0], outliers2[:, 0]),
                (outliers1[:, 1], outliers2[:, 1]), c='r', s=2)
    # plot the outliers with blue markings
    plt.plot((outliers1[:, 0], outliers2[:, 0]),
             (outliers1[:, 1], outliers2[:, 1]), mfc='r',
             c='b', lw=0.4)
    # plot the inliers with yellow markings
    plt.plot((points1[inliers][:, 0], aligned_points2[inliers][:, 0]),
             (points1[inliers][:, 1], aligned_points2[inliers][:, 1]), mfc='r',
             color='y', lw=0.4)
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
  Convert a list of succesive homographies to a 
  list of homographies to a common reference frame.
  :param H_successive: A list of M-1 3x3 homography 
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
  :param m: Index of the coordinate system towards which we would like to 
    accumulate the given homographies.
  :return: A list of M 3x3 homography matrices, 
    where H2m[i] transforms points from coordinate system i to coordinate
     system m
  """
    # Initialize list of M identity matrices
    H2m = [np.eye(3)] * (len(H_succesive) + 1)

    # Compute H2m[:m]
    for i in range(m - 1, -1, -1):
        H2m[i] = H2m[i + 1] @ H_succesive[i]

    # Compute H2m[m:]
    for i in range(m + 1, len(H2m)):
        H2m[i] = H2m[i - 1] @ np.linalg.inv(H_succesive[i - 1])

    # Normalize homographies
    H2m = [H / H[2, 2] for H in H2m]
    return H2m


def compute_bounding_box(homography, w, h):
    """
  computes bounding box of warped image under homography, without actually
   warping the image
  :param homography: homography
  :param w: width of the image
  :param h: height of the image
  :return: 2x2 array, where the first row is [x,y] of the top left corner,
   and the second row is the [x,y] of the bottom right corner
  """
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    corners_post_homography = apply_homography(corners, homography)
    min_xy = corners_post_homography.min(axis=0)
    max_xy = corners_post_homography.max(axis=0)
    return np.array([min_xy, max_xy], dtype=np.int)


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    # Compute bounding box and get image corners coordinates
    h, w = image.shape
    bounding_box = compute_bounding_box(homography, w, h)
    min_x, min_y = bounding_box[0]
    max_x, max_y = bounding_box[1]

    x, y = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
    coords = np.column_stack((x.flatten(), y.flatten()))

    # Apply homography to image coordinates
    translated_coords = apply_homography(coords, np.linalg.inv(homography))
    y_prime = translated_coords[:, 1].reshape(y.shape)
    x_prime = translated_coords[:, 0].reshape(x.shape)

    # Map image values to warped coordinates using linear interpolation
    return map_coordinates(image, [y_prime, x_prime], order=1,
                           prefilter=False)


def warp_image(image, homography):
    """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
    return np.dstack(
        [warp_channel(image[..., channel], homography) for channel in
         range(3)])


def filter_homographies_with_translation(homographies,
                                         minimum_right_translation):
    """
  Filters rigid transformations encoded as homographies by the amount of
  translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which the
   transformation is discarded.
  :return: filtered homographies..
  """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
  Computes rigid transforming points1 towards points2, using least squares
  method.
  points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate
   is *x*.
  :param points1: array with shape (N,2). Holds coordinates of corresponding
  points from image 1.
  :param points2: array with shape (N,2). Holds coordinates of corresponding
  points from image 2.
  :param translation_only: whether to compute translation only. False (default)
   to compute rotation as well.
  :return: A 3x3 array with the computed homography.
  """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True
  indicates local maximum.
  """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
  Splits the image im to m by n rectangles and uses harris_corner_detector on
   each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the
  image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates
   of the ith corner points.
  """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (
            corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
  Generates panorama from a set of images.
  """

    def __init__(self, data_dir, file_prefix, num_images, bonus=False):
        """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
        self.bonus = bonus
        self.file_prefix = file_prefix
        self.files = [
            os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i
            in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], \
                               points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], \
                           points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6,
                                             translation_only)

            # Uncomment for debugging: display inliers and outliers among
            # matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 ,
            # points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs,
                                                           (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(
            self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
  combine slices from input images to panoramas.
  :param number_of_panoramas: how many different slices to take from each input
   image
  """
        if self.bonus:
            self.generate_panoramic_images_bonus(number_of_panoramas)
        else:
            self.generate_panoramic_images_normal(number_of_panoramas)

    def generate_panoramic_images_normal(self, number_of_panoramas):
        """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each
    input image
    """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate
        # system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i],
                                                          self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2,
                                    endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros(
            (number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the
        # input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None,
                              :]
            # homography warps the slice center to the coordinate system of the
            # middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in
                              self.homographies]
            # we are actually only interested in the x coordinate of each slice
            # center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :,
                                      0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(
            np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:,
                             :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) *
                                      panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros(
            (number_of_panoramas, panorama_size[1], panorama_size[0], 3),
            dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:,
                              boundaries[0] - x_offset: boundaries[
                                                            1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom,
                boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the
        # left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few ' \
                                       'images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def generate_panoramic_images_bonus(self, number_of_panoramas):
        """
    The bonus
    :param number_of_panoramas: how many different slices to take from each
     input image
    """
        pass

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()
