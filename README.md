# FireworksCheckout
This contains a pilot computer vision project to identify fireworks packages for an automatic checkout system.

# Project Overview
At my former employer, barcodes were difficult to use in a rapid fashion, so we typed in the name and used autocomplete, which was prone to error. This pilot project explores using computer vision to replace that system. This could be extended elsewhere, as a traditional conveyor belt moving everything under a camera could be convenient for grocery stores if you could get customers to stack them one item thick.
I used an existing computer vision system, YOLO-11X, as the basis of my project, as it provided good performance with acceptable resource use. I only used fourteen fireworks as this was just a trial project. It could be scaled up if it was decided to implement it.

# Features
- **Data collection** - photographs of mixed stacks of fireworks
- **Data processing** - applying several backgrounds to each photo for generalization and labeling the fireworks in Label Studio
- **Modeling** - tried several machine learning models including R-CNN and its variants, Retinanet and its variants, and several other versions of YOLO, and YOLO-11X offered the best performance with reasonable resource use. Used mAP50-95 as my main metric during training.
- **Evaluation** - determined the number of orders with errors associated, both with a background trained on and a novel one

# Data
The dataset consists of photos of 14 unique fireworks, taken at various angles and under different lighting conditions. Backgrounds were replaced with generic storefronts to enhance generalizability. Labels were created in both YOLO and YOLO-OBB formats, with the latter being more effective for non-rectangular packaging.

Note: This dataset was generated for demonstration purposes and is not intended for production use. In a real-world scenario, photos would need to be collected in the store environment using consistent setups.

I only put a handful of files in because of space limitations. I included five images showing training file backgrounds, and two test images, one of which is determined accurately and the other of which is not.

# Methodology
After taking and labeling the photos, I removed the background and put in five sets of background per image. I was able to reuse labels because I did not change the image at all. I then sorted the first 34 images into 30 training and 4 validation photos. After training, I took another 7 photos for testing. The numbers are a little odd compared to usual machine learning ratios and probably should have been more numerous. I added augmentations including mix-up, rotations, mosaics, etc.

I trained using YOLO-11X as it offered a good compromise between computing resources and performance. I did this in several rounds with increasing resolution: 640x640, 960x960, 1280x1280 and 1600x1600. The photos of the fireworks were all 4000x3000. I was limited to 1600x1600 by VRAM.

My training metric was mAP50-95. My overall target was percent of orders with errors in them, with each testing photo considered an order.

# Key Findings
The project met mixed success but I believe it can be improved to the point of usefulness. I tested with two backgrounds, one intended to be more difficult. On the easier one, I had 100% accuracy on my test cases. With the harder one, just 71% of orders required no changes. In the particular test image I uploaded, 3B detected an extra Thundering Rainbow. Generalizability is not required for this particular implementation of this project. If I were to implement this, I would set up a tripod with a specific camera with a flash with a set background and distance to fireworks to generate the training data separately for each store. I can even set the background to something specific if the logical one doesn't work well. I was taking photos by hand at home in an inconsistent environment, including high angles that, again, are not part of the situation where this is to be used. Some fireworks were hard to tell apart because they are very similar in packaging.

# How to Run the Project
1. Clone this repository:
   git clone https://github.com/steffenadria/FireworksCheckout
2. Install the required Python packages:
   pip install -r requirements.txt
3. Run the training file:
   python yolo_fireworks_training.py
4. Run the testing file:
   python yolo_fireworks_testing.py
   This will output the names and confidence in those names in the terminal.

# Dependencies
- Python 3.8
- Ultralytics (for YOLO implementation)
  
Feel free to reach out with questions or feedback!
- Name: Steffen Adria
- Email: steffen.adria@gmail.com
- LinkedIn: [My LinkedIn]([https://linkedin.com/in/yourname](https://www.linkedin.com/in/steffen-adria/))
