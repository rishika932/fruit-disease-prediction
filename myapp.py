
import os
import cv2
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image



# Data Directory
data_dir_fruit ='FYP_DATASET\\fruit'
data_dir_leaf ='FYP_DATASET\\leaf'
model_dir = 'models'
cnn_fruit = 'CNN-FRUIT.h5'
cnn_leaf = 'CNN-LEAF.h5'

# Parameters
img_height = 256
img_width = 256
if not os.path.exists(data_dir_leaf):
    fruit_list = ['Fruits_apple_fresh',
    'Fruits_apple_rotten',
    'Fruits_banana_fresh',
    'Fruits_banana_rotten',
    'Fruits_orange_fresh',
    'Fruits_orange_rotten']
else:
    fruit_list = os.listdir(data_dir_fruit)

if not os.path.exists(data_dir_leaf):
    leaf_list =['Leaves_apple_cedar_apple_rust',
    'Leaves_apple_healthy',
    'Leaves_apple_scab',
    'Leaves_grape_black_rot',
    'Leaves_grape_healthy',
    'Leaves_grape_leaf_blight',
    'Leaves_potato_early_blight',
    'Leaves_potato_healthy',
    'Leaves_potato_late_blight',
    'Leaves_strawberry_healthy',
    'Leaves_strawberry_leaf_scorch']
else:
    leaf_list = os.listdir(data_dir_leaf)

# Load fruit model
fruit_model_path = os.path.join(model_dir, cnn_fruit)
fruit_model = tf.keras.models.load_model(fruit_model_path)

# Load leaf model
leaf_model_path = os.path.join(model_dir, cnn_leaf)
leaf_model = tf.keras.models.load_model(leaf_model_path)

#===========FUNCTIONS=====================

def tips(x):
    #Note
    # 0 --> Leaves_apple_cedar_apple_rust
    # 1 --> Leaves_apple_healthy
    # 2 --> Leaves_apple_scab
    # 3 --> Leaves_grape_black_rot
    # 4 --> Leaves_grape_healthy
    # 5 --> Leaves_grape_leaf_blight
    # 6 --> Leaves_potato_early_blight
    # 7 --> Leaves_potato_healthy
    # 8 --> Leaves_potato_late_blight
    # 9 --> Leaves_strawberry_healthy
    # 10 --> Leaves_strawberry_leaf_scorch

    if x == 0:
        tips = '\n\nhttps://www.planetnatural.com/pest-problem-solver/plant-disease/cedar-apple-rust/'+'\n\nhttps://www.gardenia.net/guide/cedar-apple-rust'+ '\n\nhttps://gardenerspath.com/how-to/disease-and-pests/cedar-apple-rust-control/'
    elif x == 2 :
        tips = '\n\nhttps://www.epicgardening.com/apple-scab/'+'\n\nhttps://www.planetnatural.com/pest-problem-solver/plant-disease/apple-scab/'+'\n\nhttps://www.missouribotanicalgarden.org/gardens-gardening/your-garden/help-for-the-home-gardener/advice-tips-resources/pests-and-problems/diseases/scabs/apple-scab'
    elif x == 3:
        tips = '\n\nhttps://www.finegardening.com/article/eco-friendly-ways-to-control-black-rot-on-a-grape-plant'+'\n\nhttps://www.gardeningknowhow.com/edible/fruits/grapes/black-rot-grape-treatment.htm'+'\n\nhttps://grapes.extension.org/black-rot-of-grapes/'
    elif x == 5:
        tips = '\n\nhttps://www.planthealthaustralia.com.au/wp-content/uploads/2013/11/Bacterial-blight-of-grapevine-FS.pdf'+'\n\nhttps://plantvillage.psu.edu/topics/grape/infos'+'\n\nhttps://www.dpi.nsw.gov.au/biosecurity/plant/insect-pests-and-plant-diseases/bacterial-blight'
    elif x == 6:
        tips = '\n\nhttps://www.apsnet.org/edcenter/disandpath/fungalasco/pdlessons/Pages/PotatoTomato.aspx'+'\n\nhttps://www.planetnatural.com/pest-problem-solver/plant-disease/early-blight/'+'\n\nhttps://www.gardeningknowhow.com/edible/vegetables/potato/potato-early-blight-treatment.htm'
    elif x == 8:
        tips = '\n\nhttps://www.dpi.nsw.gov.au/biosecurity/plant/insect-pests-and-plant-diseases/late-blight'+'\n\nhttps://www.planetnatural.com/pest-problem-solver/plant-disease/late-blight/'+'\n\nhttps://redepapa.medium.com/how-to-control-potato-late-blight-by-using-fungicides-correctly-cbf186777723'
    elif x == 10:
        tips = '\n\nhttps://www.gardeningknowhow.com/edible/fruits/strawberry/strawberries-with-leaf-scorch.htm'+'\n\nhttps://s3.amazonaws.com/assets.cce.cornell.edu/attachments/21543/strawleafscorchfs.pdf?1490125224'+'\n\nhttps://strawberryplants.org/strawberry-leaves/'
    else:
        tips = 'Healthy Plant :)'
    tips = '<p style="font-family:serif; color:White; font-size: px;">' + tips + '</p>'
    st.markdown(tips, unsafe_allow_html=True)

def preprocess(image):
    # Preprocess the image
    image = image.convert('RGB')
    resized_image = image.resize((img_width, img_height))
    img_array = tf.keras.preprocessing.image.img_to_array(resized_image)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)  
    return img_array


def button_predict(image):
    col1, col2 = st.columns(2)
    with col1:
        # Predict button
        if st.button("LEAF"):
        # Make leaf prediction
            img_array = preprocess(image)
            leaf_prediction = leaf_model.predict(img_array)
            predicted_class_index = np.argmax(leaf_prediction, axis=-1)
            leaf_class = leaf_list[predicted_class_index[0]]
            leaf_class = leaf_class.replace("_", " ")
            leaf_class = leaf_class.replace("Leaves", "")
            leaf_class = leaf_class.split(' ', 1)[1]
            leaf_class = leaf_class.title()
            leaf_class = 'Leaf | ' + leaf_class.title()
            output = '<p style="font-family:serif; color:White; font-size: 30px;">' + leaf_class + '</p>'
            st.markdown(output, unsafe_allow_html=True)
            tips(predicted_class_index[0])
            

    with col2:
        if st.button("FRUIT"):
            # Make fruit prediction
            img_array = preprocess(image)
            fruit_prediction = fruit_model.predict(img_array)
            predicted_class_index = np.argmax(fruit_prediction, axis=-1)
            fruit_class = fruit_list[predicted_class_index[0]]
            fruit_class = fruit_class.replace("_", " ")
            fruit_class = fruit_class.replace("Fruits", "")
            fruit_class = 'Fruit | ' + fruit_class.title()
            output = '<p style="font-family:serif; color:White; font-size: 30px;">' + fruit_class + '</p>'
            st.markdown(output, unsafe_allow_html=True)
            

def set_background ():
    header = """
    <style>
        header.css-1avcm0n.e13qjvis2 
        {
            background-image: url("https://i.makeagif.com/media/12-15-2017/3NEEQI.gif");
            background-size: cover;
            background-repeat: no-repeat;
        }
    </style>
    """
    st.markdown(header, unsafe_allow_html=True)

    background = """
    <style>
        section.css-uf99v8.e1g8pov65 {
            background-image: url("https://rare-gallery.com/thumbnail/421010-nature-leaves-plants-dark.jpg");
            background-size: cover;
            background-repeat: no-repeat;
        }
    </style>
    """
    st.markdown(background, unsafe_allow_html=True)

    sidebar_back = """
    <style>
        section.css-1cypcdb.e1akgbir11 {
            background-image: url("https://e1.pxfuel.com/desktop-wallpaper/360/425/desktop-wallpaper-no-419-iphone-green-and-black.jpg");
            background-size: cover;
            background-repeat: no-repeat;
        }
    </style>
    """
    st.markdown(sidebar_back, unsafe_allow_html=True)


    role_button = """
    <style>
        section.css-1erivf3.eqdbnj015 {
            background-image: url("https://w0.peakpx.com/wallpaper/3/23/HD-wallpaper-leaves-dark-plant-carved-bush.jpg");
            background-size: cover;
            background-repeat: no-repeat;
        }
    </style>
    """
    st.markdown(role_button, unsafe_allow_html=True)


#PAGE TITLE
st.title("Plant Leaf Disease & Fruit Quality Prediction" )
st.markdown(":leaves: :apple: :green_apple: :tangerine: :banana: :watermelon: :cherries: :peach: :seedling: :lemon: :strawberry:")
note = 'Plant Leaf Disease & Fruit Quality Prediction using Image Classification. Utilizing CNN (Convolutional Neural Network) to classify the plant disease and fruit quality'
st.write(note)
st.write("")
st.write("")


#SIDEBAR - 1 
st.sidebar.title("Option :wrench: ")
#SIDEBAR RADIO BUTTON
input_option = st.sidebar.radio("\n\nSelect Image Input : ", ("Image", "Camera"))
if input_option == "Camera":
    pict = st.camera_input("Capture Real Time Image :camera_with_flash:" ,key="camera", help="Basic Camera Input Feature :)")
    if pict is not None:
        image = Image.open(pict)
        button_predict(image)
        
elif input_option == "Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        show_image = image.resize((2056, 2056))
        show_image = image.resize((2056, 2056))
        st.image(show_image, caption="Uploaded Image", use_column_width=True )
        button_predict(image)

#SIDEBAR - 2
st.sidebar.title("\n")
st.sidebar.title("Leaf :leaves:")
st.sidebar.write("Apple Healthy\n\nApple Scab\n\nApple Cedar Rust\n\nGrape Healthy\n\nGrape Black Rot\n\nPotato Healthy")
st.sidebar.write("Potato Early Blight\n\nPotato Late Blight\n\nStrawberry Healthy\n\nStrawberry Leaf Scorch")

#SIDEBAR - 3
st.sidebar.title("\n")
st.sidebar.title("Fruit :green_apple:")
st.sidebar.write("Apple :apple:\n\n Banana :banana: \n\n Orange :tangerine:")

#SIDEBAR - 4
st.sidebar.title("\n")
st.sidebar.title("About  :information_source:")
st.sidebar.info('Plant Disease & Quality Prediction \n\n CNN Custom Model\n\n Kevin Ahmadiputra | TP058396 | Computer Science(DA) \n\n Asia Pacific University | 2020 - 2023')

set_background()




