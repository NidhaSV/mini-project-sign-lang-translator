import os
import pickle
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow import keras
from segmentation import segment_hand


DEFAULT_DATASET_PATH = 'sign_dataset.ds'
DEFAULT_MODEL_PATH = 'asl_classifier.keras'
LABEL_CLASS_MAP = { 0: "", 1: "Hello", 2: "Thanks", 3: "Yes", 4: "No", 5: "Help", 6:'Very Good', 7:'G', 8:'H', 9:'I',10:'J',
                11:'K', 12:'L', 13:'M', 14:'N', 15:'O', 16:'P', 17:"Q", 18:'R', 19:'S', 20:'T',
                21:'U', 22:'V', 23:'W', 24:'X', 25:'Y', 26:'Z' }
X = 0
Y = 1
INPUT_SIZE = 128
MODE_WINDOW_MOVE = 1
# Initial position of capture window
win_x, win_y, win_w, win_h = (24, 24, 310, 310)
img_h, img_w = (0, 0)
data_points = []
target_labels = []
model = None
if os.path.exists(DEFAULT_MODEL_PATH):
    print(f"Loading existing model from '{DEFAULT_MODEL_PATH}'")
    model = keras.models.load_model(DEFAULT_MODEL_PATH)
else:
    print("No model saved at '{DEFAULT_MODEL_PATH}'")



def handleArrowKeys(key, mode):
    if mode == MODE_WINDOW_MOVE:
        handleArrowKeysLabelChange(key)
    # We may add more modes to change training params gaussian_window, fine_tune_c etc


def handleArrowKeysWindowMove(key):
    global win_x, win_y, win_w, win_h
    if key == 2490368:  # Up arrow key
        win_y = win_y - 1 if win_y>0 else 0
    elif key == 2621440:  # Down arrow key
        win_y = win_y + 1 if win_y<(img_h-win_h) else img_h-win_h
    elif key == 2424832:  # Left arrow key
        win_x = win_x - 1 if win_x>0 else 0
    elif key == 2555904:  # Right arrow key
        win_x = win_x + 1 if win_x<(img_w-win_w) else img_w-win_w

def handleArrowKeysLabelChange(key):
    global train_label
    if key == 2424832:  # Left arrow key
        if train_label > 0 :
            train_label -= 1
    elif key == 2555904:  # Right arrow key
        if train_label < 26 :
            train_label += 1
    
    print("Going to train ", LABEL_CLASS_MAP[train_label])

def captureDataPoint(preprocessed_img, train_label):
    global data_points, target_labels
    resized=cv2.resize(preprocessed_img,(INPUT_SIZE,INPUT_SIZE))
    target_labels.append(train_label)
    data_points.append(resized)
    print(f"Captured data point {len(data_points)} for label {LABEL_CLASS_MAP[train_label]}")


def saveDatasetToFile(data_points, target_labels):
    file_path = '' #input(f"Path to save data set?({DEFAULT_DATASET_PATH}):")
    if file_path == '':
        file_path = DEFAULT_DATASET_PATH
    try:
        # Open the file in binary mode for writing
        with open(file_path, 'wb') as file:
            # Serialize and write the variable to the file
            pickle.dump((data_points, target_labels), file)
            label_count = len(set(target_labels))
            print(f"Saved {len(data_points)} data points with {label_count} labels to path: {file_path}")
    except Exception as ex:
        print("Failed to save dataset:", ex)


def loadDatasetFromFile():
    file_path = '' # input(f"Path to load data set?({DEFAULT_DATASET_PATH}):")
    if file_path == '':
        file_path = DEFAULT_DATASET_PATH
    try:
        # Open the file in binary mode for reading
        with open(file_path, 'rb') as file:
            # Deserialize and retrieve the variable from the file
            data, labels = pickle.load(file)
            label_set = set(labels)
            print(f"Loaded {len(data)} data points with {len(label_set)} labels from path: {file_path}")
            print("Labels are: ", label_set)
            return (data, labels)
    except Exception as ex:
        print("Failed to load dataset:", ex)


def buildNeuralNetwork(label_count):
    from keras.api.models import Sequential
    from keras.api.layers import Convolution2D
    from keras.api.layers import MaxPooling2D
    from keras.api.layers import Flatten
    from keras.api.layers import Dense , Dropout

    newModel = Sequential()
    # First convolution layer and pooling
    newModel.add(Convolution2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 1), activation='relu'))
    newModel.add(MaxPooling2D(pool_size=(2, 2)))
    # Second convolution layer and pooling
    newModel.add(Convolution2D(32, (3, 3), activation='relu'))
    # input_shape is going to be the pooled feature maps from the previous convolution layer
    newModel.add(MaxPooling2D(pool_size=(2, 2)))
    # input_shape is going to be the pooled feature maps from the previous convolution layer

    # Flattening the layers
    newModel.add(Flatten())

    # Adding a fully connected layer
    newModel.add(Dense(units=128, activation='relu'))
    newModel.add(Dropout(0.40))
    newModel.add(Dense(units=96, activation='relu'))
    newModel.add(Dropout(0.40))
    newModel.add(Dense(units=64, activation='relu'))
    newModel.add(Dense(units=label_count, activation='softmax')) # softmax for more than 2

    # Compiling the CNN - # categorical_crossentropy for more than 2
    newModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    newModel.summary()
    return newModel


def trainModel():
    from keras.api.callbacks import ModelCheckpoint
    from sklearn.model_selection import train_test_split
    from matplotlib import pyplot as plt

    global data_points, target_labels, model
    label_set = set(target_labels)
    label_count = len(label_set)
    print(f"Training for {label_count} labels: {label_set}")

    # Normalize the datapoints
    dataset = np.array(data_points)/255.0
    print(f"Changing dataset shape to ({dataset.shape[0]},{INPUT_SIZE},{INPUT_SIZE},{1})")
    dataset = np.reshape(dataset, (dataset.shape[0],INPUT_SIZE,INPUT_SIZE,1))
    target_labels_np = np.array(target_labels)
    encoded_targets = to_categorical(target_labels_np)
    
    train_data,test_data,train_target,test_target=train_test_split(dataset,encoded_targets,test_size=0.2)
    
    print("Dataset Shapes:")
    print(f"train_data:{train_data.shape},test_data:{test_data.shape},train_target:{train_target.shape},test_target:{test_target.shape}")

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # Step 1 - Building the CNN

    # Initializing the CNN
    classifier = buildNeuralNetwork(label_count) if model is None else model
    
    # Step 2 - Preparing the train/test data and training the model
    checkpoint = ModelCheckpoint(DEFAULT_MODEL_PATH, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    history=classifier.fit(train_data, train_target, shuffle=True, epochs=20, callbacks=[checkpoint], validation_split=0.3)
    model = classifier
    print("Test Results:", classifier.evaluate(test_data,test_target))
    
    N = 20
    H=history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('evaluation.png')
    # Serialize the model to disk
    print("[INFO] Saving the model to: ", DEFAULT_MODEL_PATH)
    classifier.save(DEFAULT_MODEL_PATH)
    print(f"Training completed")


def recognizeSign(preprocessed_img):
    global model
    resized = cv2.resize(preprocessed_img,(INPUT_SIZE,INPUT_SIZE))
    normalized = resized/255.0
    reshaped = np.reshape(normalized,(1,INPUT_SIZE,INPUT_SIZE,1))
    result = model.predict(reshaped)
    print("Predicted label: ", result)
    print("ArgMax=", np.argmax(result,axis=1))
    label=np.argmax(result, axis=1)[0]
    if label is not None:
        print(f"Sign detected as {LABEL_CLASS_MAP[label]}")
        return label
    return None


train_label = 0

'''
Main function
'''
def main():
    global win_x, win_y, win_w, win_h, img_h, img_w
    global train_label, data_points, target_labels

    inp = 0 # input("Which label to train? Valid labels are 0-26 (0): ")
    train_label = 0 if inp == '' else int(inp)
    print("Going to train ", LABEL_CLASS_MAP[train_label])

    source = cv2.VideoCapture(0)
    visual_threshold = 70
    gaussian_window = 11
    fine_tune_c = 2

    if not source.isOpened():
        raise Exception("Unable to open Camera. Please check any other application is using camera.")

    ret,live_img = source.read()            

    img_h, img_w = live_img.shape[:2]
    while(True):
        ret,live_img = source.read()
        
        # frame_p1 = (win_x, win_y)
        # frame_p2 = (win_x + win_w , win_y + win_h)
        # cv2.rectangle(live_img,frame_p1,frame_p2,COLOR_GREEN,2)
        # crop_img = gray[frame_p1[Y]:frame_p2[Y], frame_p1[X]:frame_p2[X]]
        # cv2.putText(live_img, "Training Frame", (win_x, win_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
        crop_img = segment_hand(live_img)
        if crop_img is not None:
            # Clean up the image data for training
            crop_gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(crop_gray,(5,5),2)
            th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,gaussian_window,fine_tune_c)
            ret, preprocessed = cv2.threshold(th3, visual_threshold, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)            
            cv2.imshow("Preprocessed", preprocessed)
        # else:
        #     print("No hand detected.")

        cv2.imshow('Live View',live_img)

        key=cv2.waitKeyEx(100)
        if(key==27):#press Esc. to exit
            break
        elif(key==ord('C') or key==ord('c')):#pressed C
            captureDataPoint(preprocessed, train_label)

        elif(key==3014656):#pressed DELETE
            deleteN = input(f"Delete how many?(Enter a number or Just press enter to delete all):")
            if deleteN == '':
                data_points = []
                target_labels = []
            else:
                data_points = data_points[:-int(deleteN)]
                target_labels = target_labels[:-int(deleteN)]

        elif(key==ord('R') or key==ord('r')):#pressed R
            recognizeSign(preprocessed)

        elif(key==ord('T') or key==ord('t')):#pressed T
            trainModel()

        elif(key==ord('S') or key==ord('s')):#pressed S
            saveDatasetToFile(data_points, target_labels)
        
        elif(key==ord('L') or key==ord('l')):#pressed L
            loaded_data = loadDatasetFromFile()
            if loaded_data is not None:
                loaded_data_points, loaded_labels = loaded_data
                data_points = data_points + loaded_data_points
                target_labels = target_labels + loaded_labels
            
        elif key in [2490368, 2621440, 2424832, 2555904]:  # Any arrow key
            handleArrowKeys(key, MODE_WINDOW_MOVE)

    cv2.destroyAllWindows()
    source.release()
    print("Terminated Sign training tool")


# Start the program
main()
