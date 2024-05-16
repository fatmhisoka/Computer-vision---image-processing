import numpy as np
import cv2

def normalize(img):
    """
    Normalize the image and display it.

    Args:
        img: Input image.
    """
    lmin = float(img.min())
    lmax = float(img.max())

    img_new = np.floor((img - lmin) / (lmax - lmin) * 225.0)

    return img_new.astype(np.uint8)


def equalization(img):
    """
    Perform histogram equalization on the image and display it.

    Args:
        img: Input image.
    """

    img = np.asarray(img)
    # put pixels in a 1D array by flattening out img array
    flat = img.flatten()
    hist = np.zeros(256)
    # loop through pixels and sum up counts of pixels
    for pixel in flat:
        hist[pixel] += 1

    a = iter(hist)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    cs = np.array(b)

    # numerator & denomenator
    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()

    # re-normalize the cdf
    cs = nj / N
    # cast it back to uint8 since we can't use floating point values in images
    cs = cs.astype('uint8')
    # get the value from cumulative sum for every index in flat, and set that as img_new
    img_new = cs[flat]
    # put array back into original shape since we flattened it
    img_new = np.reshape(img_new, img.shape)

    return  img_new.astype(np.uint8)



def local_thresholding(gray,slider_1,slider_2,slider_3,slider_4):
        """
        Apply thresholding to the image based on user-defined parameters.

        Args:
            image: Input image.
        """

        height, width = gray.shape  # get the height and width of the image
        # In this case we will divide the image into a 2x2 grid image
        half_height = height // 2
        half_width = width // 2

        # Getting the four section of the 2x2 image
        section_1 = gray[:half_height, :half_width]
        section_2 = gray[:half_height, half_width:]
        section_3 = gray[half_height:, :half_width]
        section_4 = gray[half_height:, half_width:]

        section_1[section_1 > slider_1] = 255
        section_1[section_1 < slider_1] = 0

        section_2[section_2 > slider_2] = 255
        section_2[section_2 < slider_2] = 0

        section_3[section_3 > slider_3] = 255
        section_3[section_3 < slider_3] = 0

        section_4[section_4 >slider_4] = 255
        section_4[section_4 < slider_4] = 0

        # Regroup the sections to form the final image
        top_section = np.concatenate((section_1, section_2), axis=1)
        bottom_section = np.concatenate((section_3, section_4), axis=1)
        final_img = np.concatenate((top_section, bottom_section), axis=0)

        return final_img


def global_thresholding(gray,slider_1):

        final_img = gray.copy()
        final_img[gray > slider_1] = 255
        final_img[gray < slider_1] = 0
        return final_img
