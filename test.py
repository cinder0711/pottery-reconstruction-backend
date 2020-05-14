import logging
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util
import torch

def startModel():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('PRUEBAS DE MODELO')
    logger.info("Primera prueba")
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.eval = True # For experimentation
    opt.dataset_mode = 'single' 
    opt.dataroot = './data/pottery/'
    logger.info("Segunda prueba")
    
    # Load 8 modelViews
    model0 = create_model(opt)      # create a model given opt.model and other options
    model0.setup(opt,'m0')               # regular setup: load and print networks; create schedulers
    model1 = create_model(opt)      # create a model given opt.model and other options
    model1.setup(opt,'m1')               # regular setup: load and print networks; create schedulers
    model2 = create_model(opt)      # create a model given opt.model and other options
    model2.setup(opt,'m2')               # regular setup: load and print networks; create schedulers
    model3 = create_model(opt)      # create a model given opt.model and other options
    model3.setup(opt,'m3')               # regular setup: load and print networks; create schedulers
    model4 = create_model(opt)      # create a model given opt.model and other options
    model4.setup(opt,'m4')               # regular setup: load and print networks; create schedulers
    model5 = create_model(opt)      # create a model given opt.model and other options
    model5.setup(opt,'m5')               # regular setup: load and print networks; create schedulers
    model6 = create_model(opt)      # create a model given opt.model and other options
    model6.setup(opt,'m6')               # regular setup: load and print networks; create schedulers
    model7 = create_model(opt)      # create a model given opt.model and other options
    model7.setup(opt,'m7')               # regular setup: load and print networks; create schedulers
    logger.info("Tercera prueba")
    if opt.eval:
        model0.eval()
        model1.eval()
        model2.eval()
        model3.eval()
        model4.eval()
        model5.eval()
        model6.eval()
        model7.eval()
        logger.info("Cuarta prueba")
    return model0,model1,model2,model3,model4,model5,model6,model7

def runModel(filename,model0,model1,model2,model3,model4,model5,model6,model7):
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.eval = True # For experimentation
    opt.dataset_mode = 'single' 
    opt.dataroot = './data/pottery/'+filename[:-4]

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    for i, data in enumerate(dataset): 
        model0.set_input(data,0)
        model1.set_input(data,1)
        model2.set_input(data,2)
        model3.set_input(data,3)
        model4.set_input(data,4)
        model5.set_input(data,5)
        model6.set_input(data,6)
        model7.set_input(data,7)

        # obtener shapememory en el model.shapememory
        f0 = model0.netG(model0.real)
        f1 = model1.netG(model1.real)
        f2 = model2.netG(model2.real)
        f3 = model3.netG(model3.real)
        f4 = model4.netG(model4.real)
        f5 = model5.netG(model5.real)
        f6 = model6.netG(model6.real)
        f7 = model7.netG(model7.real)

        shapeMemory = torch.cat((f0, f1, f2, f3, f4, f5, f6, f7), 1)

        model0.test(shapeMemory)
        model1.test(shapeMemory)
        model2.test(shapeMemory)
        model3.test(shapeMemory)
        model4.test(shapeMemory)
        model5.test(shapeMemory)
        model6.test(shapeMemory)
        model7.test(shapeMemory)

        visuals0 = model0.get_current_visuals()
        visuals1 = model1.get_current_visuals()
        visuals2 = model2.get_current_visuals()
        visuals3 = model3.get_current_visuals()
        visuals4 = model4.get_current_visuals()
        visuals5 = model5.get_current_visuals()
        visuals6 = model6.get_current_visuals()
        visuals7 = model7.get_current_visuals()

        img_fake0 = util.tensor2im(visuals0['fake'])
        img_fake1 = util.tensor2im(visuals1['fake'])
        img_fake2 = util.tensor2im(visuals2['fake'])
        img_fake3 = util.tensor2im(visuals3['fake'])
        img_fake4 = util.tensor2im(visuals4['fake'])
        img_fake5 = util.tensor2im(visuals5['fake'])
        img_fake6 = util.tensor2im(visuals6['fake'])
        img_fake7 = util.tensor2im(visuals7['fake'])

        img_fake = util.get_concat_v(img_fake0,img_fake1,img_fake2,img_fake3,img_fake4,img_fake5,img_fake6,img_fake7)
        img_fake.save('./data/pottery/send/'+filename)

        return