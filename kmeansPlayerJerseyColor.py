#Video/Frame Processing section
#Get frame from video
def get_frame(video_file, frame_index):
    """
    Args:
        video_file:     (str) path to .MP4 video file
        frame_index:    (int) query frame index
    Returns:
        frame:          (ndarray, size (y, x, 3)) video frame
                        Uses OpenCV BGR channels
    """

    video_capture = cv2.VideoCapture(video_file)
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = video_capture.read()
    if not success:
        raise ValueError(
            "Couldn't retrieve frame {0} from video {1}".format(
                frame_index,
                video_file
            )
        )

    return frame

#Extract player bounding boxes
def crop_bbox(frame, bbox):
    """
    Args:
        frame:      (ndarray, size (y, x, 3)) video frame
                    Uses OpenCV BGR channels
        bbox:       [u1, v1, u2, v2] integer array of pixel coordinates:
                    (u1, v1) upper-left point of bounding box
                    (u2, v2) lower-right point of bounding box
    Returns:
        bbox_rgb:   (ndarray, size (y, x, 3)) cropped bounding box
                    RGB channels
    """

    bbox_bgr = frame[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    bbox_rgb  = cv2.cvtColor(bbox_bgr, cv2.COLOR_BGR2RGB)

    return bbox_rgb

#Determine number of frames to iterate over
def count_frames(video_file):
    """
    Args:
        video_file:     (str) path to .MP4 video file
    Returns:
        nFrames:        (int) Number of frames in mp4
    """
    cap = cv2.VideoCapture(video_file)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    return(length)

#Get dictionary from json file                            
def read_json_dict(path2json):
    """
    Args:
        path2json:     (str) path to .MP4 json file containing player bounding boxes
    Returns:
        bb_dict:       (dict) Dictionary containing bounding boxes in each frame
    """
    # Opening JSON file                     
    f = open(path2json)
    
    # Returns JSON object as a dictionary
    bb_dict = json.load(f) 
    
    f.close()
    return(bb_dict)
                           
#Determine number of bounding boxes in frame
def count_bboxes(bb_dict,frame_index):
    """
    Args:
        bb_dict:      (dict) dictionary from json file
        frame:        (int) what frame is being processed
    Returns:
        nDetections:        (int) Number of bounding boxes in frame
    """
    
    bbs = bb_dict['frames'][frame_index]['detections']
    nDetections = len(bbs)
    #print(nDetections, " bounding boxes found in frame ", frame_index)
    return(nDetections)
 
#Extract bounding boxes for a given frame from json
def get_bb4frame(bb_dict,frame_index):
    """
    Args:
        bb_dict:      (dict) dictionary from json file
        frame:        (int) what frame is being processed
    Returns:
        nDetections:        (int) Number of bounding boxes in frame
    """
    
    bbs = bb_dict['frames'][frame_index]['detections']
    #print('These are the coordinates for all bounding boxes in frame', frame_index)
    #print(bbs)
    return(bbs)

#Extract bounding boxes for a given frame from json
def findFirstFrame(bb_dict,frame_index):
    """
    Args:
        bb_dict:      (dict) dictionary from json file
        frame:        (int) what frame is being processed
    Returns:
        firstFrame:        (int) First frame to process in video
    """
    
    firstFrame =  bb_dict['frames'][0]['frame_index']
    print('These is the first frame to process in video ', firstFrame)
    return(firstFrame)

#Extract bounding boxes for a given frame from json
def makeRectangleFromJSON(bb_dict,whichBB):
    """
    Args:
        bb_dict:      (dict) dictionary from json file
        whichBB:        (int) what bounding box is being processed
    Returns:
        x1 ,y1 ,x2 ,y2:    (tuple) tuple containing pixel coordinates for the upper-left and lower-right corners of the bounding box
    """
    x1 ,y1 ,x2 ,y2 = bb_dict[whichBB][0],bb_dict[whichBB][1],bb_dict[whichBB][2],bb_dict[whichBB][3]
    #print(x1 ,y1 ,x2 ,y2, ' These are the coordinates for bounding box ', whichBB)
    return(x1 ,y1 ,x2 ,y2)

def crop_image(image,howMuch):
    """
    Args:
        img        : (array) image of player bounding box
        howMuch    : (int) percent of image to crop (between 0 and 100)
    Returns:
        cropped_img :   (array) cropped image
    """
    val = howMuch/100
    cropped_img = image[0:int(image.shape[0]*val),0:int(image.shape[0])]
    return cropped_img

def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def KMeansTrace(path2img,clusters,whichFrame,whichBB):
    """
    Args:
        path2img   : (str) path to cropped player bounding box
        clusters   : (int) how many clusters to use for KMEANS. (2 and 3 work quite well)
        whichFrame : (int) which frame are we looking at
        whichBB    : (int) which bounding box are we looking at
    Returns:
        hexval :   (string) Hexcode for most dominant color in image. Should be the player jersey color
    """
    img = cv2.imread(path2img)
    org_img = img.copy()
    #print('Org image shape --> ',img.shape)

    #Remove green background/field from image prior to clustering 
    green = np.array([60,255,255]) #This is green in HSV
    loGreen = np.array([30,25,25]) #low green threshold
    hiGreen = np.array([60,255,255]) #Upper green threshold
    #Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, loGreen, hiGreen)
    result = img.copy()
    result[mask==255] = (255,255,255)
    result2 = result[np.all(result != 255 , axis=-1)] 
    cv2.imwrite(path2MaskBB + '\\Frame ' + str(whichFrame) + '_BB' + str(whichBB)+ '.jpg', result)

    #Convert image into a 1D array
    flat_img = np.reshape(result2,(-1,3))
    arrayLen = flat_img.shape
    if(arrayLen[0] < clusters):
        img = cv2.imread(path2img)
        flat_img = np.reshape(img,(-1,3))  
        
    #print('After Flattening shape --> ',flat_img.shape)

    #Do the clustering
    kmeans = KMeans(n_clusters=clusters, random_state=0, tol = 1e-4)
    kmeans.fit(flat_img)
    #Define the array with centroids
    dominant_colors = np.array(kmeans.cluster_centers_,dtype='uint')
    #Calculate percentages 
    percentages = (np.unique(kmeans.labels_,return_counts=True)[1])/flat_img.shape[0]
    size = len(percentages)
    #Insufficient clusters generated
    if size < clusters:
    #    print(percentages)
        print('Insufficient clusters generated. Likely due to overmasking. This seems to happen when BB has no player in it (i.e. only field). Clustering without mask')
        img = cv2.imread(path2img)
        flat_img = np.reshape(img,(-1,3))
        #Do the clustering
        kmeans = KMeans(n_clusters=clusters, random_state=0, tol = 1e-6)
        kmeans.fit(flat_img)
        #Define the array with centroids
        dominant_colors = np.array(kmeans.cluster_centers_,dtype='uint')
        #Calculate percentages 
        percentages = (np.unique(kmeans.labels_,return_counts=True)[1])/flat_img.shape[0]
        
    #Combine centroids representing dominant colors and percentages associated with each centroid into an array
    #print(percentages)
    if ((percentages[0] != percentages[1]) & (percentages[0] != percentages[2]) & (percentages[1] != percentages[2])):
        pc = zip(percentages,dominant_colors)
        pc = sorted(pc,reverse=True)
    else:
        pc = list(zip(percentages,dominant_colors))
    
    bgr_val1 = pc[0][1]
    bgr_val2 = pc[1][1]
    bgr_val3 = pc[2][1]
    rgbList1 = [bgr_val1[2], bgr_val1[1], bgr_val1[0]]
    rgbList2 = [bgr_val2[2], bgr_val2[1], bgr_val2[0]]
    rgbList3 = [bgr_val3[2], bgr_val3[1], bgr_val3[0]]
    hexval1 = rgb2hex(bgr_val1[2], bgr_val1[1], bgr_val1[0])
    hexval2 = rgb2hex(bgr_val2[2], bgr_val2[1], bgr_val2[0])
    hexval3 = rgb2hex(bgr_val3[2], bgr_val3[1], bgr_val3[0])
    return hexval1, hexval2, hexval3, rgbList1, rgbList2, rgbList3

def hex_to_rgb(hex_string):
    rgb = colors.hex2color(hex_string)
    return tuple([int(255*x) for x in rgb])
    
#Wrapper function
def getPlayerJerseyColorTrace(path2mp4,path2json,output_loc,nClusters,path2CroppedBB):
    """
    Args:
        path2mp4          : (array) image of player bounding box
        path2json   : (str) directory to save data to
        output_loc   : (int) current frame
        nClusters    : (int) Number of clusters to use for KMEANS
    Returns:
        color_array:   (array) array with dominant color at each bounding box
    """    
    start = time.time()
    #Need to determine starting frame!
    bb_dict = read_json_dict(path2json)
    firstJSONFrame = findFirstFrame(bb_dict,0)
    whichFrame     = firstJSONFrame 
    firstFrame = 0
    howManyFrames = 10
    bbsProcessed = 1
    #2 Count frames in video
    nFrames = int(count_frames(path2mp4)/2)
    print('There are ' + str(nFrames) + ' frames in the video.')
    #Define arrays/lists that will contain output of routine
    jerseyColor1    = [] #This list will contain hexcodes corresponding to dominant color in a bounding box
    jerseyColor2    = [] #This list will contain hexcodes corresponding to 2nd dominant color in a bounding box
    jerseyColor3    = [] #This list will contain hexcodes corresponding to 3rd dominant color in a bounding box
    jerseyColor1RGB = [] #This list will contain hexcodes corresponding to dominant color in a bounding box
    jerseyColor2RGB = [] #This list will contain hexcodes corresponding to 2nd dominant color in a bounding box
    jerseyColor3RGB = [] #This list will contain hexcodes corresponding to 3rd dominant color in a bounding box
    cboxName        = [] #This will contain the file name associated with the bounding being processed
    frameList       = [] #This will contain the frame associated with the bounding box being processed
    bbList          = [] #This will contain the bounding box in the frame being processed
    #nFrames = firstJSONFrame + 6*howManyFrames
    while whichFrame < nFrames:
        #3 Extract the frame
        currentframe = get_frame(path2mp4, whichFrame)
        cv2.imwrite(output_loc + '\\Frame ' + str(whichFrame) + '.jpg', currentframe)
        #Make copy of current frame that will be used to show all bounding boxes in image
        cv2.imwrite(output_loc + '\\Frame ' + str(whichFrame) + '_AllBB.jpg', currentframe)
        #Treat image as cv2 object and determine its size
        image  = cv2.imread(output_loc + '\\Frame ' + str(whichFrame) + '.jpg')
        result = cv2.imread(output_loc + '\\Frame ' + str(whichFrame) + '_AllBB.jpg')

        h1,w1,_ = image.shape
        image = cv2.resize(image, (w1, h1))

        #4 Read the json file --> KEYS = {version,id,frames}
        nbb = count_bboxes(bb_dict,firstFrame) #Number of bounding boxes in current frame
            
        bbs = get_bb4frame(bb_dict,firstFrame) #Bounding box coordinates for current frame

        #4 Draw bounding boxes from json file
        whichBB = 0
        i = 0
        for i in bbs:
            #print('############################################################################################################################################')
            print('######## Processing Frame ' + str(whichFrame) + ' BB ' + str(whichBB) + ' #########')
            #print('############################################################################################################################################')
            #Populate the frame and bounding box lists with current Frame and BB respectively 
            frameList.append(whichFrame)
            bbList.append(whichBB)
            
            x1 ,y1 ,x2 ,y2 = makeRectangleFromJSON(bbs,whichBB)
            #Generate current BB in current frame 
            currentbox = image[y1:y2,x1:x2]
            cv2.imwrite(path2RawBB + '\\Frame ' + str(whichFrame) + '_BB' + str(whichBB)+'.jpg', currentbox)
            
            #Crop image by 50% since we only care about the jersey color
            cropped_bb = currentbox.copy()
            cropped_bb = crop_image(currentbox,40)
            cv2.imwrite(path2CroppedBB + '\\Frame ' + str(whichFrame) + '_BB' + str(whichBB) + '_Cropped.jpg', cropped_bb)  
            
            #Determine player jersey color via kmeans
            path2cbbox = path2CroppedBB + '\\Frame ' + str(whichFrame) + '_BB' + str(whichBB) + '_Cropped.jpg'
            #print('Clustering Initiated...')
            jerseyColorInBB1,jerseyColorInBB2,jerseyColorInBB3, rgbList1, rgbList2, rgbList3 = KMeansTrace(path2cbbox,nClusters,whichFrame,whichBB)
            #print('Clustering Concluded...')
            if (jerseyColorInBB1 != ''):
                jerseyColor1.append(jerseyColorInBB1)
                jerseyColor2.append(jerseyColorInBB2)
                jerseyColor3.append(jerseyColorInBB3)
                jerseyColor1RGB.append(rgbList1)
                jerseyColor2RGB.append(rgbList2)
                jerseyColor3RGB.append(rgbList3)
            else: 
                jerseyColor1.append(None)
                jerseyColor2.append(None)
                jerseyColor3.append(None)
                jerseyColor1RGB.append(None)
                jerseyColor2RGB.append(None)
                jerseyColor3RGB.append(None)
                
            cbox = '\\Frame ' + str(whichFrame) + '_BB' + str(whichBB) + '_Cropped.jpg'
            cboxName.append(path2cbbox)
            
            #Draw ALL rectangles in current frame
            cv2.rectangle(result, (x1, y1),  (x2, y2), (0, 0, 255), 2)
            currentbox = result[y1:y2,x1:x2]
            whichBB += 1
            bbsProcessed += 1
            #Remove background from player bounding box image
            #bb_bg_removed = remove_nonplayer_region(cropped_bb, path2CroppedNoBG, whichFrame, whichBB, threshold1, threshold2, lb1, ub1)
            
        cv2.imwrite(path2Frames + '\\Frame ' + str(whichFrame) + '.jpg', currentframe)   
        cv2.imwrite(path2Frames_wBB + '\\Frame ' + str(whichFrame) + '_AllBB.jpg',result) 
    
        whichFrame += 6
        firstFrame += 1     
    
    print(str(bbsProcessed) + ' Bounding Boxes processed.')
    
    #Turn jerseyColor list into a pandas dataframe
    jerseyColor_df = pd.DataFrame({'File Name': cboxName, 'Jersey Color 1': jerseyColor1, 'Jersey Color 1 RGB ': jerseyColor1RGB, 'Jersey Color 2': jerseyColor2,  'Jersey Color 2 RGB ': jerseyColor2RGB, 'Jersey Color 3': jerseyColor3,  'Jersey Color 3 RGB ': jerseyColor3RGB, 'Frame ID': frameList, 'Bounding Box ID': bbList})
    #Clean up folder
    for zippath in glob.iglob(os.path.join(output_loc, '*.jpg')):
        os.remove(zippath)
    
    end = time.time()
    totTime = end - start
    print(str(totTime) + ' seconds elapsed for process to finish.')
    return jerseyColor_df

#Make image bigger
def makeBigger(img):
 
    #print('Original Dimensions : ',img.shape)
    scale_percent = 200 # percent of original size
    #width = int(img.shape[1] * scale_percent / 100)
    #height = int(img.shape[0] * scale_percent / 100)
    dim = (300, 200) #(width, height)
  
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

# empty function called when trackbar moves
def emptyFunction():
    pass

#Panel to cycle through player bounding boxes and 2 dominant colors in each BB   
def main(df):
    # blackwindow having 3 color chanels
    windowName ="Open CV Color Palette"
      
    # window name
    cv2.namedWindow(windowName) 
       
    # Define trackbar
    rows = df.shape[0]-1
    cv2.createTrackbar('BB ID', windowName, 0, rows, emptyFunction)
       
    #previousTrackbarValue = -1  # Set this to -1 so the threshold will be applied and the image displayed the first time through the loop
    # Used to open the window until press ESC key
    while(True):
        if cv2.waitKey(1) == 27:
            break
        # Which row to look at in dataframe?
        bbID = cv2.getTrackbarPos('BB ID', windowName)  
        print(bbID)
        fName = df.iloc[bbID]['File Name']
        print(fName)
        bb = cv2.imread(fName)
        bb = makeBigger(bb)
        bbsize = bb.shape
        image1 = np.zeros((bbsize[0], bbsize[1], 3), np.uint8) 
        image2 = np.zeros((bbsize[0], bbsize[1], 3), np.uint8) 
        image3 = np.zeros((bbsize[0], bbsize[1], 3), np.uint8) 
        # values of blue, green, red extracted from the dataframe
        hex_string1 = df.iloc[bbID]['Jersey Color 1']
        hex_string2 = df.iloc[bbID]['Jersey Color 2']
        hex_string3 = df.iloc[bbID]['Jersey Color 3']
        rgb1 = hex_to_rgb(hex_string1)
        blue1  = rgb1[2]
        green1 = rgb1[1]
        red1   = rgb1[0]
        
        rgb2 = hex_to_rgb(hex_string2)
        blue2  = rgb2[2]
        green2 = rgb2[1]
        red2   = rgb2[0]
        
        rgb3 = hex_to_rgb(hex_string3)
        blue3  = rgb3[2]
        green3 = rgb3[1]
        red3   = rgb3[0]
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX  
        # org
        org = (75, 50) 
        # fontScale
        fontScale = 1   
        # Blue color in BGR
        color = (255, 0, 0) 
        # Line thickness of 2 px
        thickness = 2

        image1[:] = [blue1, green1, red1]
        image2[:] = [blue2, green2, red2]
        image3[:] = [blue3, green3, red3]
        # Using cv2.putText() method
        image1 = cv2.putText(image1, hex_string1, org, font, fontScale, color, thickness, cv2.LINE_AA)
        image2 = cv2.putText(image2, hex_string2, org, font, fontScale, color, thickness, cv2.LINE_AA)
        image3 = cv2.putText(image3, hex_string3, org, font, fontScale, color, thickness, cv2.LINE_AA)
        
        # concatenate image Vertically
        verti = np.concatenate((bb, image1, image2, image3), axis=0)
        cv2.imshow(windowName, verti)
        
    cv2.destroyAllWindows()    
