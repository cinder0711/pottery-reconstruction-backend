from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.
    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=True)

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path)
        w, h = A_img.size
        A0 = A_img.crop((0,0,w,h/8))
        A1 = A_img.crop((0,h/8,w,h/4))
        A2 = A_img.crop((0,h/4,w,3*h/8))
        A3 = A_img.crop((0,3*h/8,w,h/2))
        A4 = A_img.crop((0,h/2,w,5*h/8))
        A5 = A_img.crop((0,5*h/8,w,3*h/4))
        A6 = A_img.crop((0,3*h/4,w,7*h/8))
        A7 = A_img.crop((0,7*h/8,w,h))
        A0 = self.transform(A0)
        A1 = self.transform(A1)
        A2 = self.transform(A2)
        A3 = self.transform(A3)
        A4 = self.transform(A4)
        A5 = self.transform(A5)
        A6 = self.transform(A6)
        A7 = self.transform(A7)
        return {'A0': A0,'A1': A1,'A2': A2,'A3': A3,'A4': A4,'A5': A5,'A6': A6,'A7': A7, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)