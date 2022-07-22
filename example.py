from api import Detector

# Initialize detector
detector = Detector(model_name='rapid',
                    weights_path='./weights/rapid_pL1_dark53_COCO608_Jun18_4000.ckpt',
                    use_cuda=False)
                    
# A simple example to run on a single image and plt.imshow() it
detector.detect_one(img_path='./images/Lunch2_000001.jpg',
                    input_size=1024, conf_thres=0.3,
                    visualize=True)
