from api import Detector

# Initialize detector
detector = Detector(model_name='rapid',
                    weights_path=r'D:\visual_studio_code\RAPiD\weights\rapid_pL1_dark53_COCO608_Jun16_2000.ckpt',
                    use_cuda=False)
                    
# A simple example to run on a single image and plt.imshow() it
detector.detect_one(img_path=r'D:\visual_studio_code\RAPiD\images\sample_1.jpg',
                    input_size=1024, conf_thres=0.3,
                    visualize=True)
