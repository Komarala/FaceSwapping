from __future__ import print_function

# %matplotlib inline
import matplotlib.pyplot as plt

import numpy as np
import cv2

import dlib


# Opening images

source = cv2.imread('./clinton.jpg')
source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
destination = cv2.imread('./merkel.jpg')
destination = cv2.cvtColor(destination, cv2.COLOR_BGR2RGB)

# plt.subplot(121)
# plt.imshow(source)
# plt.xticks([]), plt.yticks([]) 
# plt.title('Source')
# plt.subplot(122)
# plt.imshow(destination)
# plt.xticks([]), plt.yticks([])
# plt.title('Destination')
# plt.show()


# Face Detection

detector = dlib.get_frontal_face_detector()

source_facerect = detector(source)[0]
destination_facerect = detector(destination)[0]

print(source_facerect)
print(destination_facerect)
# [(321, 321) (692, 692)]
# [(362, 280) (734, 651)]

ax = plt.subplot(121)
plt.imshow(source)
ax.add_patch(plt.Rectangle(
    (source_facerect.left(), source_facerect.top()), # Top-left corner
    source_facerect.width(), # Width
    source_facerect.height(), # Height
    edgecolor='r', lw=1.0, fill=False))
plt.xticks([])
plt.yticks([])

ax = plt.subplot(122)
plt.imshow(destination)
ax.add_patch(
    plt.Rectangle(
        (destination_facerect.left(), destination_facerect.top()), # Top-left corner
        destination_facerect.width(), # Width
        destination_facerect.height(), # Height
        edgecolor='r', lw=1.0, fill=False))
plt.xticks([])
plt.yticks([])

plt.show()

# Landmark Detection

predictor = dlib.shape_predictor("<download corrosponding 'shape_predictor_68_face_landmarks.dat' file and give the path here>")
source_landmarks = []
for p in predictor(source, source_facerect).parts():
    source_landmarks.append([p.x, p.y])
source_landmarks = np.array(source_landmarks, dtype=np.int64)

destination_landmarks = []
for p in predictor(destination, destination_facerect).parts():
    destination_landmarks.append([p.x, p.y])
destination_landmarks = np.array(destination_landmarks, dtype=np.int64)

# Plot only the face landmarks
ax = plt.subplot(121)
plt.imshow(source)
plt.plot(source_landmarks[:, 0], source_landmarks[:, 1], '.')
plt.xticks([]); plt.yticks([])

ax = plt.subplot(122)
plt.imshow(destination)
plt.plot(destination_landmarks[:, 0], destination_landmarks[:, 1], '.')
plt.xticks([]); plt.yticks([])
plt.show()

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)


# Plot only the face landmarks used to line up the images
ax = plt.subplot(121)
plt.imshow(255*np.ones(source.shape))
plt.plot(source_landmarks[ALIGN_POINTS, 0], source_landmarks[ALIGN_POINTS, 1], '.')
plt.xticks([]); plt.yticks([])

ax = plt.subplot(122)
plt.imshow(255*np.ones(destination.shape))
plt.plot(destination_landmarks[ALIGN_POINTS, 0], destination_landmarks[ALIGN_POINTS, 1], '.')
plt.xticks([]); plt.yticks([])
plt.show()

# Affine Transformation

def transformation_from_points(points1, points2):
    points1 = np.matrix(points1).astype(np.float64)
    points2 = np.matrix(points2).astype(np.float64)
    # The translation t corresponds to the displacement of the centers of mass
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    # Normalize the mean of the points
    points1 -= c1
    points2 -= c2

    # The scaling corresponds to the ratio between the standard deviations
    s1 = np.std(points1)
    s2 = np.std(points2)
    # Normalize the variance of the points
    points1 /= s1
    points2 /= s2
    # Apply Singular Value decomposition on the correlation matrix of the points
    U, S, Vt = np.linalg.svd(points2.T * points1)
    # The R we seek is in fact the transpose of the one given by U * Vt.
    R = (U * Vt).T
    # Return the affine transformation matrix
    return np.hstack(((s1 / s2) * R, c1.T - (s1 / s2) * R * c2.T))

M = transformation_from_points(
        source_landmarks[ALIGN_POINTS],
        destination_landmarks[ALIGN_POINTS])

print('Transformation matrix between the two faces:')
print(M)
print('')

print('Translation:', M[0, 2], M[1, 2])
print('')
s = np.sqrt(M[0, 0]*M[1,1] - M[0, 1]*M[1, 0])
print('Scaling:', s)
print('')
theta = np.arccos(M[0, 0]/s)
print('Angle:', theta*180/np.pi)

# Extracting masks

OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS
]

def get_face_mask(img, landmarks):
    "Extracts a mask on an image around the important regions."
    # Create an empty mask
    mask = np.zeros(img.shape[:2], dtype=np.float64)
    # Compute the mask by computing the convex hull.
    for group in OVERLAY_POINTS:
        points = cv2.convexHull(landmarks[group])
        cv2.fillConvexPoly(mask, points, color=1)
    # Transform the mask into an image
    mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
    # Blur the mask
    mask = (cv2.GaussianBlur(mask, (11, 11), 0) > 0) * 1.0
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    return mask

# Mask over the source image
source_mask = get_face_mask(source, source_landmarks)

# Mask over the destination image
destination_mask = get_face_mask(destination, destination_landmarks)

# Apply the mask on the source image
source_masked = source * source_mask

# Apply the mask on the destination image
destination_masked = destination * destination_mask

# Plot the masks
# ax = plt.subplot(221)
# plt.imshow(source_mask)
# plt.xticks([]); plt.yticks([])
# ax = plt.subplot(222)
# plt.imshow(destination_mask)
# plt.xticks([]); plt.yticks([])
# ax = plt.subplot(223)
# plt.imshow(source_masked.astype(np.uint8))
# plt.xticks([]); plt.yticks([])
# ax = plt.subplot(224)
# plt.imshow(destination_masked.astype(np.uint8))
# plt.xticks([]); plt.yticks([])

# plt.show()

#######################################
# Warp the source image to the destination
#######################################

source_warped = cv2.warpAffine(source,
                   M,
                   (destination.shape[1], destination.shape[0]),
                   dst=None,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP
                )
# Plot the warped images
ax = plt.subplot(121)
plt.imshow(source_warped)
plt.xticks([]); plt.yticks([])
ax = plt.subplot(122)
plt.imshow(destination)
plt.xticks([]); plt.yticks([])
plt.show()

#######################################
# Warp the source mask to the destination
#######################################

# Warp the mask of the source image to the destination
warped_mask = cv2.warpAffine(source_mask,
                   M,
                   (destination.shape[1], destination.shape[0]),
                   dst=None,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP
                )

# Combine the warped source mask and the destination mask
# This avoids "missing" pixels after composition
combined_mask = np.max([destination_mask, warped_mask],axis=0)

#######################################
# Compose the warped source image with the destination using the mask
#######################################

# Crop the warped source image
source_cropped = source_warped * combined_mask

# Anti-crop the destination image
destination_cropped = destination * (1 - combined_mask)

# Add the two to compose the two images
composition =  source_cropped + destination_cropped

# plt.subplot(131)
# plt.imshow(source_cropped.astype(np.uint8))
# plt.title('Source')
# plt.xticks([]); plt.yticks([])
# plt.subplot(132)
# plt.imshow(destination_cropped.astype(np.uint8))
# plt.title('Destination')
# plt.xticks([]); plt.yticks([])
# plt.subplot(133)
# plt.imshow(composition.astype(np.uint8))
# plt.title('Composition')
# plt.xticks([]); plt.yticks([])

# plt.show()

cv2.imwrite('./composition_output.jpg', composition)

def correct_colours(source, destination, landmarks):
    "RGB color scaling correction"

    # Compute the size of the Gaussian filter by measuring the distance between the eyes
    blur_amount = 0.6*np.linalg.norm(
                      np.mean(landmarks[LEFT_EYE_POINTS], axis=0) -
                      np.mean(landmarks[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1

    # Blur the two images
    destination_blur = cv2.GaussianBlur(destination, (blur_amount, blur_amount), 0)
    source_blur = cv2.GaussianBlur(source, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    source_blur += (128 * (source_blur <= 1.0)).astype(source_blur.dtype)

    # Compute the color-corrected image
    return (source.astype(np.float64) * destination_blur.astype(np.float64) 
                    / source_blur.astype(np.float64))

#######################################
# Correct the colors
#######################################

# Correct colors in the warped image
source_warped_corrected = correct_colours(source_warped, destination, destination_landmarks)

# Plot the color-matched images
# ax = plt.subplot(121)
# plt.imshow(source_warped_corrected.astype(np.uint8))
# plt.xticks([]); plt.yticks([])
# ax = plt.subplot(122)
# plt.imshow(destination)
# plt.xticks([]); plt.yticks([])
# plt.show()

cv2.imwrite('./SWrappedCorrected_output.jpg', source_warped_corrected)

# Add the color-corrected warped image to the destination
composition_corrected = destination * (1 - combined_mask) + source_warped_corrected * combined_mask

# Normalize the image
composition_corrected = cv2.normalize(composition_corrected, None, 0.0, 255.0, cv2.NORM_MINMAX)

# To Plot the final image
# plt.figure()
# plt.imshow(composition_corrected.astype(np.uint8))
# plt.xticks([]); plt.yticks([])

# plt.show()

# Save the final image

cv2.imwrite('./CompositionCorrected_output.jpg', composition_corrected)