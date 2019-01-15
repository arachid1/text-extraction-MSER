# December 2018 - January 2019
# Ali Rachidi

# 1. Extraction of text by fragments from scanned PDF files using MSER (Maximally Stable Extremal Regions) and openCV.
# The goal is an extraction of groups of text with semantic meanings (paragraphs, titles, etc).
# Can be optimized by increasing the size of bounding rectangles at the clustering level (create_clusters function) and the size of capturing rectangles
# when saving them as separate files (merge_clusters function) in order to enclose cut/missed characters and improve Tesseract's efficiency
# 2. Recognition of text on each generated bloc of text Using Tesseract.
# One may ask why we didn't use Tesseract on the whole image, but fragmenting the text elements in a pdf helps maximize the efficiency of Tesseract.
# Indeed, Tesseract will read line by line and since structure is not uniform accross all documents, and that semantic meanings don't always go line by line,
# (what if we had a vertical paragraph of text?), it's better to directly feed text with semantic meanings into Tesseract, especially considering our next mission.
# 3. Semantic processing of text to maximize search results (using Word2vec?)
# Final Project Goal: Information retrieval of PDF files by keywords but also semantic connections to those keywords. 
# REMARKS: I attempted to use techniques of erosion/dilation of images, as well as clustering algorithms. 

import cv2
import numpy as np
from PIL import Image
import operator
import argparse
import imutils
from pdf2image import convert_from_path
import os
import subprocess

# This function converts to gray scale, a necessary step to effectively detect contours.

def generate_gray_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #setting up window

    # window = cv2.namedWindow('Grayscale image', cv2.WINDOW_NORMAL)
    # cv2.imshow('Grayscale image', gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return gray

# This function draws the contours and bounding boxes detected by our MSER objects.
# We will be manipulating them to isolate our text so it is good to be able to see them beforehand. 

def draw_contours_bboxes(gray, regions, bboxes, image):
    
    contours_image = image.copy()
    bboxes_image = image.copy()

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    # Setting up window for the image with contours

    # window = cv2.namedWindow('Contours of image without modification', cv2.WINDOW_NORMAL)

    cv2.polylines(contours_image, hulls, 1, (0, 255, 0))

    # cv2.imshow('Contours of image without modification', contours_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Setting up window for the image with bboxes

    # window = cv2.namedWindow('Bboxes of image without modification', cv2.WINDOW_NORMAL)

    for bbox in bboxes:
        x, y, w, h = bbox
        cv2.rectangle(bboxes_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # cv2.imshow('Bboxes of image without modification', bboxes_image)    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# The 3 following functions are helper functions used to eliminate bounding boxes that we won't be using since they can alterate our grouping/merging processes. 

def verify_height(height):
    return height > 30

def verify_width(width):
    return width > 30

def verify_black_percentage(path, bbox, pixelMap):
    x, y, w, h = bbox
    area = h * w
    count = 0.0
    for xcoord in range(x, x + w):
        for ycoord in range(y, y + h):
            if pixelMap[xcoord, ycoord] is not (255, 255, 255) and (pixelMap[xcoord, ycoord][0] == pixelMap[xcoord, ycoord][1]) and (pixelMap[xcoord, ycoord][1] == pixelMap[xcoord, ycoord][2]):
                count += 1
    ratio = float(count/area)
    return ratio < 0.45

# This function filtrates contours to preserve contours of text only, using the simple information that:
# - Contours of text are knowm to be very small.
# - Bounding boxes of text contain lots of black pixels (since text is black).
# It eliminates bounding boxes that have a large width and height or are below a threshold of black pixels using the 3 helper functions above. For example,
# they eliminate large bounding boxes that are formed from lines or geometric shape on a pdf document.
# The goal is to only have contours and bboxes of text, which makes it easier to enclose text that is close to each other, and most likely, has semantic logic.

def filter_bboxes(bboxes, path, image):
    bboxes_list = list()
    image_copy = image.copy()

    im = Image.open(path)
    pixelMap = im.load()

    for bbox in bboxes:
        x, y, w, h = bbox
        if [x, y, x + w, y + h] in bboxes_list:
            continue
        if verify_height(h) or verify_width(w) or verify_black_percentage(path, bbox, pixelMap): 
            continue
        bboxes_list.append([x, y, x + w, y + h])  # Create list of bounding boxes, with each bbox containing the left-top and right-bottom coordinates   
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # window = cv2.namedWindow('Bboxes of image after filtering', cv2.WINDOW_NORMAL)
    # cv2.imshow('Bboxes of image after filtering', image_copy)    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return bboxes_list

# This function creates clusters by grouping together (in a list) bounding boxes that lay within a range of height and width from any existing element in the cluster. 
# It takes arbitrary elements as the initial elements of clusters. It returns a list of lists of bounding boxes that are close to each other. 
# The actual grouping is not done by this function, but by the 'group_clusters' function. This function simply creates them. 
# The end goal is to have mini clusters formed in areas where there is text, such that they overlap.

def create_clusters(bboxes_list, height_interval = 45, width_interval = 45):
    clusters = list(list())
    clusters.append([bboxes_list[0]])
    appended = False
    count = 0
    for i in range(1, len(bboxes_list)):
        for cluster in clusters:
            if appended:
                appended = False
                break
            for element in cluster:
                if (abs(bboxes_list[i][0] - element[0]) <= width_interval) and (abs(bboxes_list[i][1] - element[1]) <= height_interval):
                    count += 1
                    cluster.append(bboxes_list[i])
                    appended = True
                    break
        clusters.append([bboxes_list[i]])
    return clusters

# This function takes the list of lists of bounding boxes generated above and for each list, it forms a rectangle from the smallest and highest coordinates.
# The goal is to group clusters that are close to each other and thus, create mini clusters of text that will overlap. 

def group_clusters(combined_bboxes, image):
    
    image_copy = image.copy()
    rectangles = []
    for group in combined_bboxes:
        x_min = min(group, key=lambda k: k[0])[0]  # Find min of x1
        x_max = max(group, key=lambda k: k[2])[2]  # Find max of x2
        y_min = min(group, key=lambda k: k[1])[1]  # Find min of y1
        y_max = max(group, key=lambda k: k[3])[3]  # Find max of y2
        rectangles.append([x_min, y_min, x_max, y_max])
        cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    # window = cv2.namedWindow('GROUPED Bboxes (by distance) of image', cv2.WINDOW_NORMAL)
    # cv2.imshow('GROUPED Bboxes (by distance) of image', image_copy)    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return rectangles

# Helper function that determines if a cluser and a rectangle overlap

def rect_overlaps(rect, cluster):
    if rect[0] > cluster[2] or rect[2] < cluster[0]:
        return False
    if rect[1] > cluster[3] or rect[3] < cluster[1]:
        return False
    return True

# What we observed when grouping bounding boxes that are close to each other is that they're not mutually exclusive since we're taking arbitrary elements as the
# initial elements of the clusters. Therefore, this function groups final clusters of all the bounding boxes that overlap. The goal is to bring together elements that
# have a semantic relation due to their promixity and when forming the final clusters, the function save each rectangle of text to a 'boxes' folder so that each 
# paragraph, title or sentence will have its own image. 

def merge_clusters(rectangles, local_directory, page, image):
    
    image_copy = image.copy()

    clusters = list(list())
    for rect in rectangles:
        matched = False
        for cluster in clusters:
            if rect_overlaps(rect, cluster):
                matched = True
                cluster[0] = min(cluster[0], rect[0])
                cluster[1] = min(cluster[1], rect[1])
                cluster[2] = max(cluster[2], rect[2])
                cluster[3] = max(cluster[3], rect[3])
        if not matched:
            clusters.append(rect)

    if not os.path.isdir('boxes/'):        
        os.makedirs('boxes/')
    counter = 1
    for cluster in clusters:
        x_min = cluster[0]  
        x_max = cluster[2]  
        y_min = cluster[1]  
        y_max = cluster[3]
        height = abs(y_max - y_min)
        width = abs(x_max - x_min)
        cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv2.imwrite('boxes/' + page.split('.')[0] + '-box{}.jpg'.format(counter), image[y_min : y_min + height, x_min : x_min + width])
        counter += 1

    # window = cv2.namedWindow('MERGED Bboxes (by distance) of image', cv2.WINDOW_NORMAL)
    # cv2.imshow('MERGED Bboxes (by distance) of image', image_copy)    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return clusters

def generate_blobs(page, local_directory):

    image = cv2.imread(page)

    # Create MSER object
    mser = cv2.MSER_create()

    # creates a gray image object
    gray = generate_gray_image(image)

    # regions are the contours detected on image (i.e, the circle inside the letter o); bboxes are the bouding boxes that can be formed from coordinates of regions
    regions, bboxes = mser.detectRegions(gray)

    # drawing contours and bounding boxes of an image to easily see them. Proof of Concept and helps visualize what we are working with. 
    draw_contours_bboxes(gray, regions, bboxes, image)

    # this function eliminates the bounding boxes that we don't need (i.e, formed by lines or large shapes) since they can alter our clustering processes
    bboxes_list = filter_bboxes(bboxes, page, image) # returns bboxes. Format of bbox: [x, y, x + width, y + height] 
                                                     # or [x1, y1, x2, y2] if (x1, y1) and (x2, y2) top-left and right bottom extremities
    print(len(bboxes_list))
    print(len(bboxes_list) is not None)

    if len(bboxes_list) is not 0:

        bboxes_list = sorted(bboxes_list, key=lambda k: (k[1], k[0])) # Sort by x1 then y1 coordinate (x and y of the left-top coordinate of box) 

        combined_bboxes = create_clusters(bboxes_list) # creates a list of lists of bounding boxes that are close to each other
        
        rectangles = group_clusters(combined_bboxes, image) # groups those bounding boxes by picking the extremities 

        clusters = merge_clusters(rectangles, local_directory, page, image) # merges combined boxes that overlap 

# this function is used to look at the content of a single contour. 
# However, I don't use it but thought it'd be useful to keep it

def isolate_contour(region, np_img, path):

    mask = np.zeros((np_img.shape[0], np_img.shape[1], 1), dtype=np.uint8)
    roi_corners = np.array(region, dtype=np.int32)

    channel_count = np_img.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, [roi_corners], ignore_mask_color)

    masked_image = cv2.bitwise_and(np_img, np_img, mask=mask)

    window = cv2.namedWindow("Image avec contenu d'un contour choisi", cv2.WINDOW_NORMAL)

    cv2.imshow("Image avec contenu d'un contour choisi", masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return True

def main():

    input_directory = 'Docs'
    output_directory = 'Processed_docs'
    docs = []
    file_names = []
    original_directory = os.getcwd() # absolute directory
    print(original_directory)

    # extraction of all pdfs in a given input_directory and converting them to pngs, one page per pngs
    for f in os.listdir(input_directory):
        file_names.append(f.split('.')[0])
        input_path = input_directory + '/' + f
        docs.append(convert_from_path(input_path))

    # saving all docs of segmented jpgs in folders carrying the names of their respective files (Structure: output_directory/file_name/1.jpg, 2.jpg etc)
    
    doc_count = 0
    to_save = ''
    for doc in docs:
        output_path = output_directory + '/' + file_names[doc_count] # for example, 'Docs_pdf/NOTE1'
        os.makedirs(output_path) 
        doc_count += 1
        page_count = 0
        for page in doc:
            page_count += 1
            to_save = output_path + '/' + str(page_count) + '.png' # Docs_pdf/NOTE1/1.jpg, Docs_pdf/NOTE1/2.jpg, Docs_pdf/NOTE1/3.jpg...
            page.save(to_save, 'PNG')

    # Now, we can parse through the folders and extract the text from the images
    for file_name in file_names:
        os.chdir(original_directory)
        local_directory = output_directory + '/' + file_name
        os.chdir(local_directory)
        pages = os.listdir('.')
        for page in pages:
           generate_blobs(page, local_directory)
        os.chdir('boxes/')
        os.system('for f in *.jpg;do convert "$f" -resize 200% -type Grayscale "$(basename "$f" .jpg).tif";done')
        os.system('for f in *.tif;do tesseract -l eng "$f" "$(basename "$f" .tif)";done')
        os.chdir(original_directory)
        os.chdir(local_directory)
        with open('text.txt', 'w+') as outfile:
            for f in os.listdir('boxes'):
                if f.endswith(".txt"):
                    with open('boxes/' + f) as infile:
                        outfile.write(infile.read())



if __name__ == "__main__":
    main()